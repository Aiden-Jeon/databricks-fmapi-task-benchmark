"""TXT-6 감정 분석 (한/영 병행).

이 태스크는 영어와 한국어 리뷰 텍스트의 감정 분석(negative=0, positive=1)을
구현한다. SST-2(영어)와 NSMC(한국어) 데이터셋을 사용하며, 부분 언어별 분할
샘플링을 통해 양 언어를 고르게 평가한다.
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import build_text_message
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import MulticlassAccumulator
from src.scoring.metrics import classification_metrics
from src.tasks.base import Task, Sample, register


@register
class Txt6Task(Task):
    """감정 분석 분류 태스크 (TXT-6)."""

    task_id: str = "TXT-6"
    kind: str = "classification"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """영어/한국어 감정 데이터셋에서 seed 고정 subset을 로드.

        n개 샘플을 언어별로 균등 분할: 약 n//2씩 영어와 한국어.
        각 언어별 컬럼명(sentence/document, label)을 자동 감지하고
        binary 0/1 레이블로 정규화한다.
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config:
            raise ValueError("config에 datasets 맵이 없음")

        datasets_map = config["datasets"]  # {en: sentiment_en, ko: sentiment_ko}
        samples = []
        sample_id = 0

        # 언어별 샘플 수 분할
        n_per_lang = max(1, n // len(datasets_map))
        remainder = n % len(datasets_map)

        for lang_idx, (lang, dataset_key) in enumerate(datasets_map.items()):
            # 각 언어에 할당할 샘플 수
            n_lang = n_per_lang + (1 if lang_idx < remainder else 0)

            # 레지스트리에서 데이터셋 메타 조회
            dataset_entry = resolve_dataset_entry(registry, dataset_key)
            hf_id = dataset_entry["hf_id"]
            split = dataset_entry.get("split", "train")
            config_name = dataset_entry.get("config")
            # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
            revision = dataset_entry.get("revision")

            # HF 데이터셋 로드 (seed 고정)
            hf_ds = load_hf_split(hf_id, split, n_lang, seed, config_name, revision)

            # 컬럼명 자동 감지: sentiment_en(sentence), sentiment_ko(document)
            col_text = self._detect_text_column(hf_ds)
            col_label = "label"  # 둘 다 "label"

            for idx, row in enumerate(hf_ds):
                text = row[col_text]
                label_raw = row[col_label]

                # binary 정규화: negative=0, positive=1
                # SST-2: label 0/1이지만 validation에는 -1이 없으므로 그대로 사용
                # NSMC: label 0/1
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

    def _detect_text_column(self, hf_ds: Any) -> str:
        """HF 데이터셋에서 텍스트 컬럼명을 자동 감지.

        일반적으로 'sentence'(SST-2) 또는 'document'(NSMC).
        """
        column_names = list(hf_ds[0].keys()) if hf_ds else []

        for col in ["sentence", "document", "text"]:
            if col in column_names:
                return col

        # 컬럼 감지 실패 시 첫 번째 문자열 컬럼 사용
        for col in column_names:
            if col != "label":
                return col

        raise ValueError(f"텍스트 컬럼을 찾을 수 없음. 컬럼: {column_names}")

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """명확한 감정 분류 프롬프트 구성.

        모델에게 "positive" 또는 "negative" 중 정확히 하나의 단어로
        응답하도록 요청한다. 한국어 샘플도 영어 지시로 통일.
        """
        text = sample.inputs["text"]

        prompt = f"""Classify the sentiment of the following text as exactly one word: "positive" or "negative".

Text: {text}

Respond with exactly one word: positive or negative"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "positive"→1, "negative"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시 첫 단어만 추출.
        한국어 응답("긍정"/"부정") 처리.
        파싱 불가 시 None 반환.
        """
        if not raw_text or not raw_text.strip():
            return None

        text_lower = raw_text.strip().lower()

        # 영어 매핑
        if "positive" in text_lower:
            return 1
        if "negative" in text_lower:
            return 0

        # 한국어 매핑 (긍정/부정)
        if "긍정" in raw_text:
            return 1
        if "부정" in raw_text:
            return 0

        # 숫자로도 시도: 1/0 또는 예/아니오 유사 표현
        if re.search(r"\b1\b|yes|true", text_lower):
            return 1
        if re.search(r"\b0\b|no|false", text_lower):
            return 0

        return None

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 메트릭 계산.

        - None 예측값 제외 (unparseable)
        - classification_metrics로 accuracy/macro_f1 계산
        - 언어별 분석 및 평가 샘플 수 포함
        """
        # None값 필터링
        valid_indices = [i for i, p in enumerate(parsed) if p is not None]

        if not valid_indices:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "n_evaluated": 0,
                "n_unparsed": len(parsed),
                "per_language": {},
            }

        preds_valid = [parsed[i] for i in valid_indices]
        golds_valid = [samples[i].reference for i in valid_indices]

        # 메트릭 계산
        metrics = classification_metrics(preds_valid, golds_valid)

        # 언어별 통계
        lang_stats = self._compute_per_language_stats(parsed, samples, valid_indices)

        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "n_evaluated": len(valid_indices),
            "n_unparsed": len(parsed) - len(valid_indices),
            "per_language": lang_stats,
        }

    def make_accumulator(self) -> "_Txt6Accumulator":
        """스트리밍 O(1) 채점기. score()와 동일(전체 accuracy/macro_f1 + per_language)."""
        return _Txt6Accumulator()

    def _compute_per_language_stats(
        self,
        parsed: list[int | None],
        samples: list[Sample],
        valid_indices: list[int],
    ) -> dict[str, Any]:
        """언어별 정확도 및 샘플 수 집계."""
        lang_data = {}

        for lang in {"en", "ko"}:
            lang_indices = [i for i in valid_indices if samples[i].lang == lang]

            if not lang_indices:
                lang_data[lang] = {
                    "n_evaluated": 0,
                    "n_unparsed": sum(1 for i in range(len(parsed)) if samples[i].lang == lang and parsed[i] is None),
                    "accuracy": None,
                }
            else:
                preds_lang = [parsed[i] for i in lang_indices]
                golds_lang = [samples[i].reference for i in lang_indices]
                metrics_lang = classification_metrics(preds_lang, golds_lang)

                lang_data[lang] = {
                    "n_evaluated": len(lang_indices),
                    "n_unparsed": sum(1 for i in range(len(parsed)) if samples[i].lang == lang and parsed[i] is None),
                    "accuracy": metrics_lang["accuracy"],
                }

        return lang_data

class _Txt6Accumulator:
    """전체 다중클래스(accuracy/macro_f1) + per_language(accuracy만) 스트리밍. score()와 동일.

    per_language 각 언어는 {n_evaluated, n_unparsed, accuracy}(유효 없으면 accuracy=None).
    """

    def __init__(self) -> None:
        self._overall = MulticlassAccumulator()
        self._lang = {"en": MulticlassAccumulator(), "ko": MulticlassAccumulator()}
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
            }
        return {
            "accuracy": o["accuracy"],
            "macro_f1": o["macro_f1"],
            "n_evaluated": o["n_evaluated"],
            "n_unparsed": o["n_unparsed"],
            "per_language": per_language,
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
    """간단한 인라인 테스트: 6샘플(영어 3, 한국어 3)로 end-to-end 실행."""
    import sys

    from src.adapters.fmapi import FMAPIClient

    # 테스트용 작은 태스크 config
    test_config = {
        "datasets": {
            "en": "sentiment_en",
            "ko": "sentiment_ko",
        }
    }

    registry = load_registry()
    task = Txt6Task(test_config, registry)

    # 6샘플 로드 (영어 3, 한국어 3)
    print("Loading 6 samples (3 en, 3 ko)...")
    samples = task.load_samples(n=6, seed=42)
    print(f"Loaded {len(samples)} samples")

    for sample in samples:
        print(f"  [{sample.sample_id}] {sample.lang}: {sample.inputs['text'][:60]}... (ref={sample.reference})")

    # 프롬프트 생성 및 모델 호출
    print("\nBuilding prompts and calling FMAPIClient...")

    try:
        with FMAPIClient(profile=_selfcheck_profile(), timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n  Sample {sample.sample_id} ({sample.lang}):")
                print(f"    Text: {sample.inputs['text'][:50]}...")

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
            print(f"Macro F1: {scores['macro_f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")

            print(f"\nPer-language breakdown:")
            for lang, stats in scores["per_language"].items():
                print(f"  {lang}:")
                print(f"    Evaluated: {stats['n_evaluated']}")
                print(f"    Accuracy: {stats['accuracy']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
