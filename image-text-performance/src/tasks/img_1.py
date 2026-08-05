"""IMG-1 이미지 캡셔닝 태스크 (generation).

이 태스크는 이미지를 주고 모델이 한 문장의 캡션을 생성하도록 한다.
- 데이터셋: yerevann/coco-karpathy (COCO 이미지 + 5개 참고 캡션)
- 메트릭: token_f1 (lexical overlap, 임시), judge (LLM-as-judge)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵
"""

from __future__ import annotations

import urllib.request
from io import BytesIO
from typing import Any

from PIL import Image

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import MeanAccumulator
from src.scoring.metrics import bertscore_f1, token_f1
from src.scoring.judge import build_judge_prompt, load_rubrics, run_judge, summarize_judge_scores
from src.tasks.base import Task, Sample, register

# BERTScore용 버퍼 상한 (TXT-5와 동일 이유 — 배치 메트릭이라 모아야 하는데 O(1) 스트리밍을
# 깨지 않도록 상한을 둔다). 캡션은 한 문장이라 메모리 부담이 없다.
BERTSCORE_MAX_PAIRS = 200


def _first_ref(reference: Any) -> str:
    """캡션 참조는 이미지당 5개다. BERTScore는 다중참조를 직접 지원하지 않고 최대값을
    쓰면 점수가 낙관적으로 치우치므로 **첫 참조**로 고정한다(token_f1은 5개 중 최고를 쓴다 —
    두 지표의 참조 정책이 다른 점은 리포트 해석 시 유의)."""
    if isinstance(reference, list):
        return str(reference[0]) if reference else ""
    return str(reference or "")


def _caption_bertscore(parsed: list[str], samples: list[Sample]) -> dict:
    """캡션 BERTScore. score()와 누적기가 같은 값을 내도록 이 함수를 공유한다."""
    pairs = [
        (str(p or ""), _first_ref(s.reference)) for p, s in zip(parsed, samples)
    ][:BERTSCORE_MAX_PAIRS]
    return bertscore_f1([c for c, _ in pairs], [r for _, r in pairs])


class _Img1Accumulator:
    """MeanAccumulator(token_f1)를 감싸고 BERTScore용 쌍을 상한까지 버퍼링한다.

    BERTScore는 배치 메트릭이라 per-sample 누적이 불가해, 러너의 스트리밍 계약
    (add/finalize)은 유지하면서 텍스트만 제한적으로 모아 finalize에서 한 번 계산한다.
    """

    def __init__(self, inner: MeanAccumulator) -> None:
        self._inner = inner
        self._pairs: list[tuple[str, str]] = []

    def add(self, parsed: Any, sample: Any) -> None:
        self._inner.add(parsed, sample)
        if len(self._pairs) < BERTSCORE_MAX_PAIRS:
            self._pairs.append((str(parsed or ""), _first_ref(sample.reference)))

    def finalize(self) -> dict[str, Any]:
        out = self._inner.finalize()
        out.update(bertscore_f1([c for c, _ in self._pairs], [r for _, r in self._pairs]))
        return out


@register
class Img1Task(Task):
    """이미지 캡셔닝 생성 태스크 (IMG-1)."""

    task_id: str = "IMG-1"
    kind: str = "generation"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """COCO-Karpathy 데이터셋에서 이미지+참고 캡션을 로드.

        yerevann/coco-karpathy는 URL을 제공하므로 해당 URL에서
        이미지를 다운로드해 PIL.Image로 변환한다.
        reference는 5개 캡션 리스트를 저장한다.
        """
        registry = load_registry()
        dataset_entry = resolve_dataset_entry(registry, "img_caption")

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "validation")
        config_name = dataset_entry.get("config")
        # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
        revision = dataset_entry.get("revision")

        # HF 데이터셋 로드
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name, revision)

        samples = []
        for idx, row in enumerate(hf_ds):
            url = row.get("url")
            sentences = row.get("sentences", [])

            if not url or not sentences:
                continue

            # URL에서 이미지 다운로드
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    image_data = response.read()
                image = Image.open(BytesIO(image_data))
                image.load()  # 즉시 로드 (lazy loading 방지)
            except Exception as e:
                # URL 접근 실패 시 스킵
                print(f"  이미지 로드 실패 (URL: {url[:50]}...): {e}")
                continue

            sample = Sample(
                sample_id=idx,
                inputs={"image": image},
                reference=sentences,  # list of 5 captions
                lang="en",
                meta={
                    "dataset": "img_caption",
                    "filename": row.get("filename", ""),
                    "cocoid": row.get("cocoid", ""),
                    "url": url,
                },
            )
            samples.append(sample)

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """이미지를 보고 한 문장의 캡션을 생성하는 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "Describe this image in one sentence."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정규화된 캡션 텍스트로 파싱.

        단순히 앞뒤 공백을 제거하고 반환.
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 캡션을 참고 캡션과 비교해 token_f1 + BERTScore 계산.

        각 샘플마다 생성된 캡션과 5개 참고 캡션 중 최고 F1을 선택.
        BERTScore는 실제로 계산한다(예전엔 "deferred (torch 미설치)"를 하드코딩해
        torch가 설치된 환경에서도 값이 안 나왔다). 불가하면 그 이유가 값으로 들어온다.
        캡션은 참조가 5개라 BERTScore는 **첫 참조**를 쓴다(다중참조 BERTScore는
        bert_score가 직접 지원하지 않고, 최대값을 쓰면 점수가 낙관적으로 치우친다).
        """
        if not parsed or not samples:
            return {
                "caption_token_f1": 0.0,
                "n_evaluated": 0,
            }

        valid_count = 0
        total_f1 = 0.0

        for pred, sample in zip(parsed, samples):
            if not pred:
                continue

            # reference는 list of captions
            references = sample.reference
            if not isinstance(references, list):
                references = [references]

            # 최고 token_f1 선택 (best match among references)
            best_f1 = 0.0
            for ref in references:
                f1 = token_f1(pred, ref, lang=sample.lang)
                best_f1 = max(best_f1, f1)

            total_f1 += best_f1
            valid_count += 1

        mean_f1 = total_f1 / valid_count if valid_count > 0 else 0.0

        out: dict[str, Any] = {
            "caption_token_f1": float(mean_f1),
            "n_evaluated": valid_count,
        }
        out.update(_caption_bertscore(parsed, samples))
        return out

    def make_accumulator(self) -> MeanAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일(caption_token_f1 + n_evaluated).

        빈 예측은 None 반환 → 평균·개수 모두에서 제외(score()의 valid_count 의미와 동일).
        n_evaluated = 유효(비어있지 않은) 예측 수 → count_all=False.
        """
        def value_fn(pred, sample):
            if not pred:
                return None
            refs = sample.reference if isinstance(sample.reference, list) else [sample.reference]
            best = 0.0
            for ref in refs:
                best = max(best, token_f1(pred, ref, lang=sample.lang))
            return best

        return _Img1Accumulator(
            MeanAccumulator(out_key="caption_token_f1", value_fn=value_fn, count_all=False)
        )

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = "databricks-gemini-3-1-pro",
    ) -> dict[str, Any]:
        """LLM-as-judge로 캡션 품질 평가.

        생성된 캡션과 참고 캡션을 텍스트로만 비교 (이미지는 불필요).
        각 샘플마다 judge 모델을 호출해 1-5 점수를 얻는다.
        """
        if not parsed or not samples:
            return {
                "judge_score_mean": 0.0,
                "judge_scores": [],
                "n_evaluated": 0,
            }

        try:
            rubrics = load_rubrics()
            rubric = rubrics.get(self.task_id, {
                "name": "Image Captioning",
                "description": "Evaluate image captions for accuracy, completeness, and clarity.",
                "anchors": {
                    "1": "Inaccurate or missing caption",
                    "2": "Partially correct caption",
                    "3": "Acceptable caption with minor issues",
                    "4": "Good caption with minor omissions",
                    "5": "Excellent, complete, and accurate caption",
                }
            })
        except Exception as e:
            print(f"Warning: 루브릭 로드 실패: {e}")
            rubric = {}

        judge_scores = []

        for pred, sample in zip(parsed, samples):
            if not pred:
                judge_scores.append(None)
                continue

            references = sample.reference
            if not isinstance(references, list):
                references = [references]

            # 최고 점수의 참고 캡션을 선택
            best_reference = references[0] if references else ""

            # Judge 프롬프트 구성
            judge_prompt = build_judge_prompt(
                task_id=self.task_id,
                question="Describe this image in one sentence.",
                reference=best_reference,
                candidate=pred,
                rubric=rubric,
            )

            # 호출·파싱·실패로그는 run_judge가 담당(max_tokens=JUDGE_MAX_TOKENS 공용).
            # 이 태스크가 max_tokens=256을 쓰던 동안 캡션 판정 근거가 길어 30/30 파싱 실패 →
            # 리포트에 judge_mean=0.0이 찍혔다(정량 caption_token_f1은 정상이었음).
            judge_scores.append(
                run_judge(judge_client, judge_endpoint, judge_prompt, self.task_id, sample.sample_id)
            )

        # 유효 점수만 평균. 전부 실패면 mean=None(0.0으로 채우면 "최악 판정"처럼 오독된다).
        agg = summarize_judge_scores(judge_scores)
        return {
            "judge_score_mean": agg["judge_mean"],
            "judge_scores": judge_scores,
            "n_evaluated": agg["n_judged"],
            "n_judge_failed": agg["n_judge_failed"],
        }



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
    """간단한 end-to-end 테스트: 2샘플로 캡셔닝 실행."""
    import sys

    print("=== IMG-1 캡셔닝 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "img_caption"}}
    task = Img1Task(task_config, registry)

    # 2샘플 로드
    print("Loading 2 samples...")
    try:
        samples = task.load_samples(n=2, seed=42)
        print(f"Loaded {len(samples)} samples\n")
    except Exception as e:
        print(f"Error loading samples: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not samples:
        print("No samples loaded!", file=sys.stderr)
        sys.exit(1)

    for sample in samples[:1]:
        print(f"[Sample {sample.sample_id}]")
        print(f"  Image: PIL shape {sample.inputs['image'].size}")
        print(f"  Reference captions: {sample.reference[:2]}...")
        print()

    # 프롬프트 생성 및 모델 호출
    print("Calling FMAPI (claude-opus-5)...\n")

    try:
        with FMAPIClient(profile=_selfcheck_profile(), timeout_seconds=30) as client:
            parsed_outputs = []

            for i, sample in enumerate(samples):
                print(f"[Sample {sample.sample_id}]")

                # 프롬프트 구성
                messages = task.build_prompt(sample)

                # FMAPI 호출
                response = client.chat(
                    endpoint="databricks-claude-opus-5",
                    messages=messages,
                    max_tokens=256,
                    extra_params={"thinking": {"type": "disabled"}},
                )

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"  Raw response: {response.text[:100]}...")
                print(f"  Parsed: {parsed[:80]}...")
                print(f"  Reference[0]: {sample.reference[0][:80]}...")
                print()

            # 채점
            print("\n=== Scoring ===\n")
            scores = task.score(parsed_outputs, samples)

            print(f"Caption token_f1: {scores['caption_token_f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Notes: {scores['notes']}")

            # Judge 점수 (선택사항)
            print("\n=== Judge Scoring (with Gemini) ===\n")
            judge_scores = task.judge_scores(parsed_outputs, samples, client)
            print(f"Judge mean score: {judge_scores['judge_score_mean']:.2f}")
            print(f"Judge scores: {judge_scores['judge_scores']}")
            print(f"Evaluated by judge: {judge_scores['n_evaluated']}/{len(samples)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
