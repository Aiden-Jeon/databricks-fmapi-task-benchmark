"""TXT-1 문서(PDF) 이해 QA (Document QA).

DocVQA로 **문서 텍스트를 읽고 답하는 능력**을 평가한다. TXT-1은 텍스트 태스크
(is_vision=False)라 페이지 이미지는 쓰지 않고, 데이터셋이 제공하는 **OCR 텍스트**를
문서 컨텍스트로 프롬프트에 넣는다.

데이터셋(`nielsr/docvqa_1200_examples`, registry의 `doc_vqa`) 구조:
- query: 질문. **dict**(de/en/es/fr/it) — en만 사용
- words: 페이지 OCR 토큰 list → 공백 결합해 문서 컨텍스트로
- answers: 정답 문자열 리스트(표기 변형 포함, 예: ['485', '$485'])
- image: PIL Image (사용 안 함)

**회귀 주의 (2026-08-05 수정)**: 이전에는 `HuggingFaceM4/DocumentVQA`를 썼는데 그 미러엔
OCR 컬럼이 **없어** `row.get("words")`가 항상 None이 되고, 코드가 조용히 "질문만" 프롬프트로
폴백했다. 그 결과 30/30 샘플이 문서 없이 전송돼 TXT-1이 문서 이해력이 아니라 **무문맥 QA**를
측정했다(token_f1 0.006 등 바닥 점수의 원인). 그래서 지금은 OCR 컨텍스트가 없으면
**조용히 폴백하지 않고 명확히 실패**한다(CLAUDE.md의 "합성/조용한 폴백 금지" 원칙과 동일).

채점: ANLS(DocVQA 공식) + Token-F1 + exact_match + LLM-judge.
ANLS는 편집거리 기반이라 표기 차이('485' vs '$485')에 관용적이면서 무관한 답은 τ=0.5로 절단한다.
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_text_message, FMAPIClient
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import MultiMeanAccumulator
from src.scoring.metrics import anls, token_f1, exact_match
from src.scoring.judge import build_judge_prompt, load_rubrics, run_judge, summarize_judge_scores
from src.tasks.base import Task, Sample, register

# OCR 컨텍스트 상한(문자). DocVQA 페이지 OCR은 실측 median ~1.0k, max ~3.7k라 대개
# 그대로 들어간다. 이상치가 프롬프트·비용을 폭증시키지 않도록만 잘라 둔다(TXT-5와 같은 방식).
MAX_CONTEXT_CHARS = 8000


@register
class Txt1Task(Task):
    """문서(PDF) 이해 QA 태스크 (TXT-1)."""

    task_id: str = "TXT-1"
    kind: str = "qa"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """DocVQA 데이터셋에서 seed 고정 subset을 로드.

        각 샘플:
        - inputs: {"question": 영어 질문, "context": 페이지 OCR 텍스트}
        - reference: 정답 문자열 리스트(표기 변형 포함)

        컨텍스트가 만들어지지 않으면 **예외를 던진다**. 문서 없이 질문만 보내면
        태스크가 측정하려는 것(문서 이해력)이 아닌 것을 재게 되므로, 조용한 폴백보다
        실패가 낫다(모듈 docstring의 회귀 주의 참고).
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config or "en" not in config["datasets"]:
            raise ValueError("config에 datasets.en (doc_vqa)가 없음")

        dataset_key = config["datasets"]["en"]
        dataset_entry = resolve_dataset_entry(registry, dataset_key)

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "test")
        config_name = dataset_entry.get("config")
        # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
        revision = dataset_entry.get("revision")

        # HF 데이터셋 로드 (seed 고정)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name, revision)
        if not hf_ds:
            raise ValueError(f"TXT-1: {hf_id}[{split}]에서 샘플을 로드하지 못했습니다")

        samples = []
        skipped = 0
        for row in hf_ds:
            question = self._extract_question(row)
            ocr_context = self._extract_ocr_context(row)
            answers_list = self._extract_answers(row)

            # 세 요소 중 하나라도 없으면 채점이 무의미 → 이 행은 건너뛰고 아래에서 집계 검증
            if not question or not ocr_context or not answers_list:
                skipped += 1
                continue

            if len(ocr_context) > MAX_CONTEXT_CHARS:
                ocr_context = ocr_context[:MAX_CONTEXT_CHARS]

            samples.append(
                Sample(
                    sample_id=len(samples),
                    inputs={
                        "question": question,
                        "context": ocr_context,
                    },
                    reference=answers_list,
                    lang="en",
                    meta={
                        "dataset": dataset_key,
                        "context_chars": len(ocr_context),
                    },
                )
            )

        if not samples:
            raise ValueError(
                f"TXT-1: {hf_id}[{split}] {len(hf_ds)}행에서 유효 샘플이 0개입니다"
                f"(질문·OCR·정답 중 누락으로 {skipped}행 스킵). "
                f"컬럼 스키마가 바뀌었을 수 있습니다: {sorted(hf_ds[0].keys())}"
            )
        if skipped:
            print(f"  [TXT-1] {skipped}행 스킵(질문·OCR·정답 누락), {len(samples)}샘플 사용")

        return samples

    @staticmethod
    def _extract_question(row: dict[str, Any]) -> str:
        """질문 추출. `query`가 언어별 dict(de/en/es/fr/it)면 en을 쓴다.

        예전 미러는 `question`(str)이었고 현재 미러는 `query`(dict)라 둘 다 받는다.
        dict를 문자열로 잘못 쓰면 질문이 통째로 깨지므로 명시적으로 분기한다.
        """
        raw = row.get("question") or row.get("query")
        if isinstance(raw, dict):
            return str(raw.get("en") or "").strip()
        return str(raw or "").strip()

    @staticmethod
    def _extract_ocr_context(row: dict[str, Any]) -> str:
        """페이지 OCR 텍스트를 문서 컨텍스트 문자열로 결합.

        `words`는 토큰 list(현 미러) 또는 {좌표: 토큰} dict일 수 있다. 레이아웃은
        보존되지 않으므로 공백으로 이어 평문 문서로 만든다.
        """
        for col in ("words", "ocr_text", "ocr", "text", "context"):
            raw = row.get(col)
            if not raw:
                continue
            if isinstance(raw, str):
                return " ".join(raw.split())
            if isinstance(raw, dict):
                return " ".join(str(v) for v in raw.values() if v)
            if isinstance(raw, list):
                return " ".join(str(w) for w in raw if w)
        return ""

    @staticmethod
    def _extract_answers(row: dict[str, Any]) -> list[str]:
        """정답 리스트 정규화. 빈 문자열은 제외."""
        raw = row.get("answers")
        if raw is None:
            raw = row.get("answer")
            # 이 미러의 `answer`는 {'text': ..., 'match_score': ...} dict
            if isinstance(raw, dict):
                raw = raw.get("text")
        if raw is None:
            return []
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        return [str(a).strip() for a in items if str(a).strip()]

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """문서 QA 프롬프트 구성: 문서 OCR 텍스트 + 질문.

        load_samples가 컨텍스트를 보장하므로 "질문만" 분기는 없다(그 폴백이 무문맥 QA
        버그의 원인이었다 — 모듈 docstring 참고).

        정답이 문서에 적힌 짧은 문자열이라, 서술형 답변이 나오면 ANLS·exact_match가
        표기 차이로 과소평가된다. 그래서 "짧게, 문서 표기 그대로"를 명시한다.
        """
        question = sample.inputs["question"]
        context = sample.inputs["context"]

        prompt = f"""Based on the following document text, answer the question.

Document text:
{context}

Question: {question}

Answer with only the exact value from the document — no explanation, no full sentence.

Answer:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정답으로 파싱.

        Token-F1과 정확도 계산을 위해 원문 그대로 반환 (정규화 없음).
        """
        return raw_text.strip()

    # per-sample 채점 함수. score()와 make_accumulator()가 **같은 함수를 공유**해
    # 두 경로의 수치가 어긋날 수 없게 한다(누적기는 score()와 bit-identical해야 함).
    # 각 값은 정답 리스트 중 최고값이고, 빈 예측은 0.0(평균 분모에는 포함).
    @staticmethod
    def _score_anls(pred: str, sample: Sample) -> float:
        """DocVQA 공식 메트릭. 다중정답 max·τ 절단은 anls()가 처리."""
        return anls(pred, sample.reference) if pred else 0.0

    @staticmethod
    def _score_token_f1(pred: str, sample: Sample) -> float:
        if not pred:
            return 0.0
        return max((token_f1(pred, g, "en") for g in sample.reference), default=0.0)

    @staticmethod
    def _score_exact_match(pred: str, sample: Sample) -> float:
        if not pred:
            return 0.0
        return max((exact_match(pred, g) for g in sample.reference), default=0.0)

    def _metric_fns(self) -> dict[str, Any]:
        """대표 메트릭 우선순위대로: anls(공식) → token_f1 → exact_match."""
        return {
            "anls": self._score_anls,
            "token_f1": self._score_token_f1,
            "exact_match": self._score_exact_match,
        }

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측을 집계해 ANLS·Token-F1·exact_match 계산.

        각 샘플별로 여러 정답 중 최고 점수를 취한 뒤 전체 평균.
        """
        fns = self._metric_fns()
        if not parsed or not samples:
            return {k: 0.0 for k in fns} | {"n_evaluated": 0}

        sums = {k: 0.0 for k in fns}
        n = 0
        for pred, sample in zip(parsed, samples):
            for k, fn in fns.items():
                sums[k] += fn(pred, sample)
            n += 1

        out: dict[str, Any] = {k: (sums[k] / n if n else 0.0) for k in fns}
        out["n_evaluated"] = n
        return out

    def make_accumulator(self) -> MultiMeanAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일 키·수치(같은 per-sample 함수 공유)."""
        return MultiMeanAccumulator(self._metric_fns())

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = "databricks-gemini-3-1-pro",
    ) -> dict[str, Any]:
        """LLM 판사를 사용한 정성적 평가.

        각 샘플에 대해 판사 모델을 호출해 1-5 점수를 얻는다.
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

        # TXT-1용 rubric (없으면 generic QA 사용)
        if "TXT-1" in rubrics:
            rubric = rubrics["TXT-1"]
        else:
            # Fallback: generic QA rubric
            rubric = {
                "name": "Document QA",
                "description": "Extract accurate answers from document text",
                "anchors": {
                    "1": "Answer is unrelated or completely incorrect",
                    "2": "Answer reflects only partial document content or core is distorted",
                    "3": "Answer accurately reflects most of document content but with minor errors or omissions",
                    "4": "Answer is nearly identical to reference with only minor phrasing differences",
                    "5": "Answer is identical or equivalent to reference with perfect accuracy",
                }
            }

        judge_scores = []
        for pred, sample in zip(parsed, samples):
            question = sample.inputs["question"]
            context = sample.inputs.get("context", "")
            # reference_list 중 첫 번째를 참고정답으로 사용
            reference = sample.reference[0] if sample.reference else ""

            # Judge prompt 구성. 문서 컨텍스트는 load_samples가 보장하지만, judge가
            # 재구성 경로(러너의 _run_judge_streaming)로 들어올 때를 대비해 방어적으로 조립.
            q_for_judge = f"Document text: {context}\n\nQuestion: {question}" if context else f"Question: {question}"
            judge_prompt = build_judge_prompt(
                task_id="TXT-1",
                question=q_for_judge,
                reference=reference,
                candidate=pred,
                rubric=rubric,
            )

            # 호출·파싱·실패로그는 run_judge가 담당. 실패는 None(집계 제외) — 옛 코드의
            # "3점으로 메우기"는 judge 평균을 조용히 오염시켰다.
            judge_scores.append(
                run_judge(judge_client, judge_endpoint, judge_prompt, self.task_id, sample.sample_id)
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
    """End-to-end 테스트: 3개 DocumentVQA 샘플로 token_f1, exact_match, judge 점수 계산."""
    import sys

    # 테스트용 태스크 config
    test_config = {
        "datasets": {
            "en": "doc_vqa",
        }
    }

    registry = load_registry()
    task = Txt1Task(test_config, registry)

    # 3샘플 로드
    print("=" * 70)
    print("TXT-1 Document QA Task - End-to-End Test")
    print("=" * 70)
    print("Loading 3 DocumentVQA samples...")
    samples = task.load_samples(n=3, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    for sample in samples:
        q_preview = sample.inputs["question"][:60]
        ocr_preview = sample.inputs["context"][:60].replace("\n", " ")
        print(f"[샘플 {sample.sample_id}]")
        print(f"  질문: {q_preview}...")
        print(f"  OCR 텍스트({sample.meta['context_chars']}자): {ocr_preview}...")
        print(f"  정답: {sample.reference}")
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

            # 수치 채점 (ANLS, Token-F1, Exact Match)
            print("\n" + "=" * 70)
            print("Computing ANLS / Token-F1 / Exact Match...")
            print("=" * 70)
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== 수치 점수 ===")
            print(f"ANLS (DocVQA 공식): {scores['anls']:.4f}")
            print(f"Token-F1: {scores['token_f1']:.4f}")
            print(f"Exact Match: {scores['exact_match']:.4f}")
            print(f"평가 샘플 수: {scores['n_evaluated']}")

            # 누적기(스트리밍 경로)가 score()와 같은 수치를 내는지 확인 — 러너는 누적기를 쓴다
            acc = task.make_accumulator()
            for pred, sample in zip(parsed_outputs, samples):
                acc.add(pred, sample)
            acc_scores = acc.finalize()
            same = all(
                abs(acc_scores[k] - scores[k]) < 1e-12 for k in ("anls", "token_f1", "exact_match")
            )
            print(f"누적기 일치: {'✓' if same else '✗ 불일치! ' + str(acc_scores)}")

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
