"""TXT-8 비속어/혐오성 텍스트 검출 (한/영 병행).

이 태스크는 영어와 한국어 텍스트의 혐오성/비속어 이진 분류를 구현한다.
- 영어: Jigsaw 독성 댓글 분류 데이터셋 (toxic=1, clean=0)
- 한국어: APEACH 한국어 혐오성 텍스트 데이터셋 (class=1: 혐오성, class=0: 정상)

부분 언어별 샘플 분할을 통해 양 언어를 고르게 평가한다.
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import build_text_message
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import BinaryAccumulator
from src.scoring.metrics import binary_metrics
from src.tasks.base import Task, Sample, register


@register
class Txt8Task(Task):
    """비속어/혐오성 텍스트 이진 분류 태스크 (TXT-8)."""

    task_id: str = "TXT-8"
    kind: str = "binary"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """영어/한국어 혐오성 데이터셋에서 seed 고정 subset을 로드.

        n개 샘플을 언어별로 균등 분할: 약 n//2씩 영어와 한국어.
        각 언어별 컬럼명(comment_text/text, toxic/class)을 자동 감지하고
        binary 0/1 레이블로 정규화한다.
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config:
            raise ValueError("config에 datasets 맵이 없음")

        datasets_map = config["datasets"]  # {en: toxicity_en, ko: toxicity_ko}
        samples = []
        sample_id = 0

        # 언어별 샘플 수 분할
        n_per_lang = max(1, n // len(datasets_map))
        remainder = n % len(datasets_map)

        for lang_idx, (lang, dataset_key) in enumerate(datasets_map.items()):
            # 각 언어에 할당할 샘플 수
            n_lang = n_per_lang + (1 if lang_idx < remainder else 0)

            # 레지스트리에서 데이터셋 메타 조회 (registry가 단일 소스)
            dataset_entry = resolve_dataset_entry(registry, dataset_key)
            hf_id = dataset_entry["hf_id"]
            split = dataset_entry.get("split", "train")
            config_name = dataset_entry.get("config")

            # HF 데이터셋 로드 (seed 고정)
            hf_ds = load_hf_split(hf_id, split, n_lang, seed, config_name)

            # 컬럼명 자동 감지
            col_text = self._detect_text_column(hf_ds, lang)
            col_label = self._detect_label_column(hf_ds, lang)

            for idx, row in enumerate(hf_ds):
                text = row[col_text]
                label_raw = row[col_label]

                # None 또는 NaN 값 스킵
                if label_raw is None or text is None:
                    continue

                # binary 정규화: toxic=1, clean=0
                label_int = int(label_raw)

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"text": text},
                    reference=label_int,
                    lang=lang,
                    meta={
                        "dataset": dataset_key,
                        "source_idx": idx,
                    },
                )
                samples.append(sample)
                sample_id += 1

        return samples

    def _detect_text_column(self, hf_ds: Any, lang: str) -> str:
        """HF 데이터셋에서 텍스트 컬럼명을 자동 감지.

        영어: 'comment_text' (Jigsaw dataset)
        한국어: 'text' (APEACH dataset)
        """
        column_names = list(hf_ds[0].keys()) if hf_ds else []

        # 언어별 우선순위
        priority_cols = {
            "en": ["comment_text", "text"],
            "ko": ["text", "comment_text"],
        }

        for col in priority_cols.get(lang, ["text", "comment_text"]):
            if col in column_names:
                return col

        # 컬럼 감지 실패 시 첫 번째 문자열 컬럼 사용
        for col in column_names:
            if col not in ["toxic", "class", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "id"]:
                return col

        raise ValueError(f"텍스트 컬럼을 찾을 수 없음. 컬럼: {column_names}")

    def _detect_label_column(self, hf_ds: Any, lang: str) -> str:
        """HF 데이터셋에서 레이블 컬럼명을 자동 감지.

        영어: 'toxic' (Jigsaw dataset)
        한국어: 'class' (APEACH dataset)
        """
        column_names = list(hf_ds[0].keys()) if hf_ds else []

        # 언어별 우선순위
        priority_cols = {
            "en": ["toxic", "class"],
            "ko": ["class", "toxic"],
        }

        for col in priority_cols.get(lang, ["toxic", "class"]):
            if col in column_names:
                return col

        raise ValueError(f"레이블 컬럼을 찾을 수 없음. 컬럼: {column_names}")

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """명확한 혐오성/비속어 검출 프롬프트 구성.

        모델에게 "toxic" 또는 "clean" 중 정확히 하나의 단어로
        응답하도록 요청한다. 한국어 샘플도 영어 지시로 통일.
        이는 임상적·중립적인 콘텐츠 관리 벤치마크 프롬프트이다.
        """
        text = sample.inputs["text"]

        prompt = f"""You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".

Text: {text}

Respond with exactly one word: toxic or clean"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "toxic"→1, "clean"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시 첫 단어만 추출.
        한국어 응답("유해"/"정상") 처리.
        파싱 불가 시 None 반환.
        """
        if not raw_text or not raw_text.strip():
            return None

        text_lower = raw_text.strip().lower()

        # 영어 매핑
        if "toxic" in text_lower:
            return 1
        if "clean" in text_lower:
            return 0

        # 한국어 매핑 (유해/정상)
        if "유해" in raw_text:
            return 1
        if "정상" in raw_text:
            return 0

        # 숫자로도 시도: 1/0
        if re.search(r"\b1\b", text_lower):
            return 1
        if re.search(r"\b0\b", text_lower):
            return 0

        return None

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 메트릭 계산.

        - None 예측값 제외 (unparseable)
        - binary_metrics로 accuracy/f1 계산
        - 혼동 행렬(confusion matrix) 포함
        - 언어별 분석 및 평가 샘플 수 포함
        """
        # None값 필터링
        valid_indices = [i for i, p in enumerate(parsed) if p is not None]

        if not valid_indices:
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "confusion_matrix": {
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 0,
                },
                "n_evaluated": 0,
                "n_unparsed": len(parsed),
                "per_language": {},
            }

        preds_valid = [parsed[i] for i in valid_indices]
        golds_valid = [samples[i].reference for i in valid_indices]

        # 메트릭 계산
        metrics = binary_metrics(preds_valid, golds_valid)

        # 언어별 통계
        lang_stats = self._compute_per_language_stats(parsed, samples, valid_indices)

        return {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "n_evaluated": len(valid_indices),
            "n_unparsed": len(parsed) - len(valid_indices),
            "per_language": lang_stats,
        }

    def make_accumulator(self) -> "_Txt8Accumulator":
        """스트리밍 O(1) 채점기. score()와 동일(전체 + per_language en/ko)."""
        return _Txt8Accumulator()

    def _compute_per_language_stats(
        self,
        parsed: list[int | None],
        samples: list[Sample],
        valid_indices: list[int],
    ) -> dict[str, Any]:
        """언어별 정확도·F1·샘플 수 집계."""
        lang_data = {}

        for lang in {"en", "ko"}:
            lang_indices = [i for i in valid_indices if samples[i].lang == lang]

            if not lang_indices:
                lang_data[lang] = {
                    "n_evaluated": 0,
                    "n_unparsed": sum(1 for i in range(len(parsed)) if samples[i].lang == lang and parsed[i] is None),
                    "accuracy": None,
                    "f1": None,
                }
            else:
                preds_lang = [parsed[i] for i in lang_indices]
                golds_lang = [samples[i].reference for i in lang_indices]
                metrics_lang = binary_metrics(preds_lang, golds_lang)

                lang_data[lang] = {
                    "n_evaluated": len(lang_indices),
                    "n_unparsed": sum(1 for i in range(len(parsed)) if samples[i].lang == lang and parsed[i] is None),
                    "accuracy": metrics_lang["accuracy"],
                    "f1": metrics_lang["f1"],
                }

        return lang_data

class _Txt8Accumulator:
    """전체 + 언어별(en/ko) 이진 채점을 스트리밍으로. score()와 동일 dict 반환.

    per_language의 각 언어는 {n_evaluated, n_unparsed, accuracy, f1}이며,
    유효 샘플이 없으면 accuracy/f1=None (score()의 _compute_per_language_stats와 동일).
    """

    def __init__(self) -> None:
        self._overall = BinaryAccumulator(include_confusion=True)  # class_balance 미사용
        self._lang = {"en": BinaryAccumulator(include_confusion=True),
                      "ko": BinaryAccumulator(include_confusion=True)}
        self._lang_unparsed = {"en": 0, "ko": 0}

    def add(self, parsed: Any, sample: Sample) -> None:
        lang = getattr(sample, "lang", "en")
        if parsed is None and lang in self._lang_unparsed:
            self._lang_unparsed[lang] += 1
        self._overall.add(parsed, sample)
        sub = self._lang.get(lang)
        if sub is not None:
            sub.add(parsed, sample)

    def finalize(self) -> dict[str, Any]:
        o = self._overall.finalize()
        per_language = {}
        for lang, sub in self._lang.items():
            f = sub.finalize()
            has = f["n_evaluated"] > 0
            per_language[lang] = {
                "n_evaluated": f["n_evaluated"],
                "n_unparsed": self._lang_unparsed[lang],
                "accuracy": f["accuracy"] if has else None,
                "f1": f["f1"] if has else None,
            }
        return {
            "accuracy": o["accuracy"],
            "f1": o["f1"],
            "confusion_matrix": o["confusion_matrix"],
            "n_evaluated": o["n_evaluated"],
            "n_unparsed": o["n_unparsed"],
            "per_language": per_language,
        }


if __name__ == "__main__":
    """간단한 인라인 테스트: 6샘플(영어 3, 한국어 3)로 end-to-end 실행."""
    import sys

    from src.adapters.fmapi import FMAPIClient

    # 테스트용 작은 태스크 config
    test_config = {
        "datasets": {
            "en": "toxicity_en",
            "ko": "toxicity_ko",
        }
    }

    registry = load_registry()
    task = Txt8Task(test_config, registry)

    # 6샘플 로드 (영어 3, 한국어 3)
    print("Loading 6 samples (3 en, 3 ko)...")
    samples = task.load_samples(n=6, seed=42)
    print(f"Loaded {len(samples)} samples")

    for sample in samples:
        text_preview = sample.inputs["text"][:60].replace("\n", " ")
        print(f"  [{sample.sample_id}] {sample.lang}: {text_preview}... (ref={sample.reference})")

    # 프롬프트 생성 및 모델 호출
    print("\nBuilding prompts and calling FMAPIClient...")

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                text_preview = sample.inputs["text"][:50].replace("\n", " ")
                print(f"\n  Sample {sample.sample_id} ({sample.lang}):")
                print(f"    Text: {text_preview}...")

                # FMAPI 호출
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=64,
                    extra_params={"reasoning_effort": "none"},
                )

                print(f"    Raw response: {response.text}")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"    Parsed: {parsed} (expected: {sample.reference})")

            # 채점
            print("\n\nScoring...")
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== Results ===")
            print(f"Accuracy: {scores['accuracy']:.3f}")
            print(f"F1: {scores['f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")
            print(f"Confusion matrix: {scores['confusion_matrix']}")

            print(f"\nPer-language breakdown:")
            for lang, stats in scores["per_language"].items():
                print(f"  {lang}:")
                print(f"    Evaluated: {stats['n_evaluated']}")
                print(f"    Accuracy: {stats['accuracy']}")
                print(f"    F1: {stats['f1']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
