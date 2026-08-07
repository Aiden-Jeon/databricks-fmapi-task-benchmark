"""TXT-2 표(엑셀) 이해 QA (Table QA).

이 태스크는 WikiTableQuestions 데이터셋을 사용하여 테이블 기반 질의응답을 평가한다.
TXT-2는 텍스트 태스크(is_vision=False)로, 마크다운 형식의 테이블 텍스트와 질문을
입력으로 받아 테이블에서 정보를 추출하여 정답을 생성한다.

WikiTableQuestions 구조:
- question: 테이블에 대한 자연어 질문
- answers: 정답 문자열 리스트 (숫자 또는 텍스트)
- table_md: 마크다운 형식의 테이블
- table: 구조화된 테이블 dict (header, rows)

정확도(숫자/문자 값 매칭, whitespace/case 관대) + Token-F1을 계산.
LLM 판사(judge)를 통한 정성적 평가도 지원한다.
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import build_text_message, FMAPIClient
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import MultiMeanAccumulator
from src.scoring.metrics import token_f1
from src.scoring.judge import build_judge_prompt, load_rubrics, run_judge_batch, summarize_judge_scores
from src.tasks.base import Task, Sample, register


def _normalize_answer(answer: str) -> str:
    """정답 정규화: 공백 제거, 소문자, 숫자 표준화."""
    answer = answer.strip().lower()
    # 숫자 후행 공백 제거 (e.g., "2004 " → "2004")
    answer = " ".join(answer.split())
    return answer


def _accuracy_match(pred: str, gold_list: list[str]) -> float:
    """정확도: 정규화된 정답과의 부분 일치 (최대값).

    gold_list 중 하나와 정확히 매칭되면 1.0, 아니면 0.0.
    """
    pred_norm = _normalize_answer(pred)

    for gold in gold_list:
        gold_norm = _normalize_answer(gold)
        if pred_norm == gold_norm:
            return 1.0

    return 0.0


@register
class Txt2Task(Task):
    """표(엑셀) 이해 QA 태스크 (TXT-2)."""

    task_id: str = "TXT-2"
    kind: str = "qa"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """WikiTableQuestions 데이터셋에서 seed 고정 subset을 로드.

        WikiTableQuestions 구조:
        - question: 자연어 질문
        - answers: 정답 리스트 (str)
        - table_md: 마크다운 형식 테이블
        - table: 구조화된 테이블 (header, rows)

        마크다운 테이블을 입력으로 사용.

        **split 주의**: 이 미러는 train이 10행뿐이고 실데이터는 test(18486행)에 있다.
        registry의 `table_qa.split: test`가 정본이며, 요청 수보다 적게 로드되면 예외를 던진다
        (조용히 작은 n으로 채점하지 않는다 — CLAUDE.md "조용한 폴백 금지").
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config or "en" not in config["datasets"]:
            raise ValueError("config에 datasets.en (table_qa)가 없음")

        dataset_key = config["datasets"]["en"]
        dataset_entry = resolve_dataset_entry(registry, dataset_key)

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "test")
        config_name = dataset_entry.get("config")
        # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
        revision = dataset_entry.get("revision")

        # HF 데이터셋 로드 (seed 고정)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name, revision)

        # 요청보다 적게 로드되면 실패시킨다. 이 미러는 train이 10행뿐이라(실데이터는 test)
        # 예전엔 30샘플 요청에도 조용히 n=10으로 채점되고, 리포트에는 다른 태스크와 같은
        # 무게로 표시됐다 — 표본이 작아진 걸 눈치챌 방법이 없었다.
        if len(hf_ds) < n:
            raise ValueError(
                f"TXT-2: {hf_id}[{split}]에서 {n}개 요청에 {len(hf_ds)}개만 로드됨 "
                f"(split 확인 필요 — 이 미러는 train 10행 / test 18486행)"
            )

        samples = []
        for sample_id, row in enumerate(hf_ds):
            question = row.get("question", "")
            answers_raw = row.get("answers", [])

            # answers 정규화: 문자열 리스트로
            if isinstance(answers_raw, list):
                answers_list = [str(a) for a in answers_raw]
            else:
                answers_list = [str(answers_raw)]

            # 테이블 텍스트 추출: table_md (마크다운) 우선, 없으면 table dict 변환
            table_text = row.get("table_md", "")

            if not table_text:
                # table dict를 마크다운으로 변환
                table_dict = row.get("table", {})
                if table_dict:
                    table_text = self._table_dict_to_markdown(table_dict)

            sample = Sample(
                sample_id=sample_id,
                inputs={
                    "table": table_text,
                    "question": question,
                },
                reference=answers_list,  # 여러 정답 리스트
                lang="en",
                meta={
                    "dataset": dataset_key,
                    "source_id": row.get("id", ""),
                },
            )
            samples.append(sample)

        return samples

    def _table_dict_to_markdown(self, table_dict: dict[str, Any]) -> str:
        """구조화된 테이블 dict를 마크다운 문자열로 변환.

        table_dict 구조:
        {
            "header": ["col1", "col2", ...],
            "rows": [["val1", "val2", ...], ...],
            "name": "table_name" (optional)
        }
        """
        if not table_dict:
            return ""

        header = table_dict.get("header", [])
        rows = table_dict.get("rows", [])

        if not header:
            return ""

        # 마크다운 테이블 구성
        md_lines = []

        # 헤더 행
        md_lines.append("| " + " | ".join(str(h) for h in header) + " |")

        # 구분자 행
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # 데이터 행
        for row in rows:
            # row가 리스트 또는 dict일 수 있음
            if isinstance(row, dict):
                values = [str(row.get(h, "")) for h in header]
            else:
                values = [str(v) for v in row]

            # 컬럼 개수 맞추기
            while len(values) < len(header):
                values.append("")

            md_lines.append("| " + " | ".join(values[: len(header)]) + " |")

        return "\n".join(md_lines)

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """테이블 QA 프롬프트 구성.

        **짧은 답만 요구한다 (2026-08-05 수정).** 이전 프롬프트는 "answer the question
        accurately" + "Answer:"뿐이라 모델이 근거를 곁들인 산문으로 답했다. 정답이
        `['4']`인데 출력이 "Looking at the table, the parishes founded in the 1800s are:
        1. **St Mary** (Bacup) – 1852 …"(135토큰)여서 문자열 일치가 0점이 됐다.
        그 결과 accuracy 0.0~0.1인데 judge는 3.7~4.4로 갈렸다 — 표를 못 읽은 게 아니라
        형식이 안 맞은 것이다. WikiTableQuestions의 정답은 짧은 셀값이므로 그 형식을 명시한다.
        """
        table = sample.inputs["table"]
        question = sample.inputs["question"]

        prompt = f"""Based on the following table, answer the question.

Table:
{table}

Question: {question}

Output ONLY the answer value(s) copied from the table — no explanation, no
reasoning, no sentence. The answer is short (a cell value such as a name,
number, or date). If the answer has multiple values, separate them with ", ".

Answer:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답에서 답만 뽑는다.

        프롬프트로 짧은 답을 요구하지만 모델이 여전히 설명을 붙일 수 있어(특히 reasoning
        모델), 마지막 방어선으로 산문에서 답을 추출한다. 순서:
        1) "Answer:" 뒤가 있으면 그 부분만
        2) 여러 줄이면 마지막 비어있지 않은 줄(결론이 뒤에 온다)
        3) 마크다운 장식(**, 백틱, 끝 마침표) 제거
        원문을 그대로 두면 정답 문자열과 절대 일치하지 않아 accuracy가 0에 고정된다.
        """
        text = (raw_text or "").strip()
        if not text:
            return ""

        # 1) 여러 줄이면 마지막 비어있지 않은 줄부터 본다 — 설명 뒤에 결론이 오는 형태 대응.
        #    목록 항목(1. / - )으로 끝나면 결론이 아니라 열거이므로 전체를 유지한다.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) > 1 and not re.match(r"^([-*•]|\d+[.)])\s", lines[-1]):
            text = lines[-1]

        # 2) "Answer:" 표기가 있으면 그 뒤만 남긴다. 1)보다 뒤에 둬야 한다 — 마지막 줄이
        #    "**Answer: 2 matches** were played…" 형태로 오는 경우가 실측으로 흔하다.
        m = re.search(r"(?:final\s+)?answer\s*[:：]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()

        # 3) 마크다운 장식·후행 구두점 정리. `**`는 짝이 안 맞게 잘려 올 수 있어(예:
        #    "2 matches** were played") 쌍 제거 후 남은 것도 그냥 지운다.
        text = text.strip().strip("`").strip()
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = text.replace("**", "").strip()
        text = re.sub(r"[.]$", "", text).strip()
        return text

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 정확도와 Token-F1 계산.

        각 샘플별로 여러 정답 중 최고 점수를 취한다.
        """
        if not parsed or not samples:
            return {
                "accuracy": 0.0,
                "token_f1": 0.0,
                "n_evaluated": 0,
            }

        accuracy_scores = []
        token_f1_scores = []

        for pred, sample in zip(parsed, samples):
            if not pred:
                # 빈 예측
                accuracy_scores.append(0.0)
                token_f1_scores.append(0.0)
                continue

            # 여러 정답 중 최고 점수
            reference_list = sample.reference  # list[str]

            # 정확도: 값 매칭 (관대한 정규화)
            max_accuracy = _accuracy_match(pred, reference_list)
            accuracy_scores.append(max_accuracy)

            # Token-F1: 모든 정답 중 최고값
            max_f1 = 0.0
            for gold in reference_list:
                f1 = token_f1(pred, gold, "en")
                max_f1 = max(max_f1, f1)

            token_f1_scores.append(max_f1)

        return {
            "accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            "token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0,
            "n_evaluated": len(parsed),
        }

    def make_accumulator(self) -> MultiMeanAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일(accuracy, token_f1, n_evaluated).

        빈 예측 → 0.0, n_evaluated=len(parsed). accuracy는 _accuracy_match(관대한 정규화),
        token_f1은 reference_list 중 최고값.
        """
        def _acc(pred, sample):
            if not pred:
                return 0.0
            return _accuracy_match(pred, sample.reference)

        def _f1(pred, sample):
            if not pred:
                return 0.0
            return max((token_f1(pred, g, "en") for g in sample.reference), default=0.0)

        return MultiMeanAccumulator({"accuracy": _acc, "token_f1": _f1})

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = "databricks-gemini-3-1-pro",
        *,
        judge_concurrency: int = 1,
    ) -> dict[str, Any]:
        """LLM 판사를 사용한 정성적 평가.

        각 샘플에 대해 판사 모델을 호출해 1-5 점수를 얻는다.
        judge_concurrency>1이면 호출을 병렬로 겹쳐 실행한다(점수·순서는 동일).
        """
        if not parsed or not samples:
            return {
                "judge_scores": [],
                "judge_mean": 0.0,
                "n_judged": 0,
            }

        # Rubric 로드
        try:
            rubrics = load_rubrics("config/judge_rubrics.yaml")
        except FileNotFoundError:
            rubrics = {}

        # TXT-2용 rubric (없으면 generic QA 사용)
        if "TXT-2" in rubrics:
            rubric = rubrics["TXT-2"]
        else:
            # Fallback: generic table QA rubric
            rubric = {
                "name": "Table QA",
                "description": "Extract accurate cell values from tables",
                "anchors": {
                    "1": "Extracted value does not match table or is completely wrong",
                    "2": "Extracted value partially matches with errors in key values",
                    "3": "Extracted value mostly correct but with minor cell errors or format issues",
                    "4": "Extracted value nearly accurate with only minor format differences",
                    "5": "Extracted value is accurate and perfectly matches reference cell values",
                }
            }

        # judge 프롬프트를 샘플 순서대로 구성한 뒤 병렬로 채점한다(입력 순서 보존).
        # 실패는 None(집계 제외) — 옛 코드의 "3점으로 메우기"는 judge 평균을 조용히 오염시켰다.
        items = []
        for pred, sample in zip(parsed, samples):
            question = sample.inputs["question"]
            table = sample.inputs["table"]
            # reference_list 중 첫 번째를 참고정답으로 사용
            reference = sample.reference[0] if sample.reference else ""
            judge_prompt = build_judge_prompt(
                task_id="TXT-2",
                question=f"Table:\n{table}\n\nQuestion: {question}",
                reference=reference,
                candidate=pred,
                rubric=rubric,
            )
            items.append((judge_prompt, sample.sample_id))

        judge_scores = run_judge_batch(
            judge_client, judge_endpoint, items, self.task_id, max_workers=judge_concurrency
        )
        return summarize_judge_scores(judge_scores)



def _selfcheck_profile() -> str:
    """자체점검(__main__)용 프로파일. config/models.yaml을 읽어 하드코딩을 피한다.

    프로파일을 코드에 박아두면 다른 워크스페이스에서 이 파일을 직접 실행할 때
    엉뚱한 곳으로 호출·과금된다. 러너 본체는 `--profile`/config를 쓰므로 여기도 맞춘다.
    """
    try:
        from src.config import load_models_config

        return load_models_config().profile
    except Exception:
        return "DEFAULT"

if __name__ == "__main__":
    """End-to-end 테스트: 3개 WikiTableQuestions 샘플로 accuracy, token_f1, judge 점수 계산."""
    import sys

    # 테스트용 태스크 config
    test_config = {
        "datasets": {
            "en": "table_qa",
        }
    }

    registry = load_registry()
    task = Txt2Task(test_config, registry)

    # 3샘플 로드
    print("=" * 70)
    print("TXT-2 Table QA Task - End-to-End Test")
    print("=" * 70)
    print("Loading 3 WikiTableQuestions samples...")
    samples = task.load_samples(n=3, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    for sample in samples:
        q_preview = sample.inputs["question"][:60]
        table_preview = sample.inputs["table"][:80].replace("\n", " ")
        ref_preview = sample.reference[0][:40] if sample.reference else "(no ref)"
        print(f"[샘플 {sample.sample_id}]")
        print(f"  질문: {q_preview}...")
        print(f"  테이블: {table_preview}...")
        print(f"  정답(첫번째): {ref_preview}...")
        print()

    # 프롬프트 생성 및 모델 호출
    print("=" * 70)
    print("Calling FMAPIClient (databricks-gpt-5-6-sol)...")
    print("=" * 70)

    try:
        with FMAPIClient(profile=_selfcheck_profile(), timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n[샘플 {sample.sample_id}]")
                print(f"  질문: {sample.inputs['question'][:50]}...")

                # FMAPI 호출 (databricks-gpt-5-6-sol + minimal reasoning)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=128,
                    extra_params={"reasoning_effort": "none"},
                )

                print(f"  모델 응답: {response.text[:80]}...")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                if sample.reference:
                    print(f"  정답(첫번째): {sample.reference[0]}")
                print(f"  파싱된 예측: {parsed}")

            # 수치 채점 (Accuracy, Token-F1)
            print("\n" + "=" * 70)
            print("Computing Accuracy and Token-F1...")
            print("=" * 70)
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== 수치 점수 ===")
            print(f"Accuracy: {scores['accuracy']:.4f}")
            print(f"Token-F1: {scores['token_f1']:.4f}")
            print(f"평가 샘플 수: {scores['n_evaluated']}")

            # Judge 호출
            print("\n" + "=" * 70)
            print("Calling Judge (databricks-gemini-3-1-pro)...")
            print("=" * 70)

            try:
                with FMAPIClient(profile=_selfcheck_profile(), timeout_seconds=60) as judge_client:
                    judge_result = task.judge_scores(
                        parsed_outputs,
                        samples,
                        judge_client,
                        judge_endpoint="databricks-gemini-3-1-pro",
                    )

                    print(f"\n=== Judge 점수 (1-5) ===")
                    print(f"개별 점수: {judge_result['judge_scores']}")
                    print(f"평균 점수: {judge_result['judge_mean']:.2f}")
                    print(f"평가 샘플 수: {judge_result['n_judged']}")
            except Exception as e:
                print(f"Judge 호출 실패: {e}")

            print("\n" + "=" * 70)
            print("테스트 완료")
            print("=" * 70)

    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
