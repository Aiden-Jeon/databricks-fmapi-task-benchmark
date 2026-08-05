"""IMG-5 사람 포함 여부 이진 분류 (binary).

이 태스크는 이미지에서 사람(person) 객체의 존재 여부를 판별한다.
- 데이터셋: detection-datasets/coco (validation split)
- 레이블: binary (1=사람 있음, 0=사람 없음)
- 메트릭: accuracy, f1 (binary_metrics)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵

COCO 데이터셋의 objects.category config에서:
- 클래스 ID 0 = person (COCO 공식 클래스 순서)
- 클래스 ID 1-79 = 기타 객체들

사람 판별: 객체 카테고리에 ID 0(person)이 있으면 1, 없으면 0.
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.accumulators import BinaryAccumulator
from src.scoring.metrics import binary_metrics
from src.tasks.base import Task, Sample, register


@register
class Img5Task(Task):
    """사람 포함 여부 이진 분류 태스크 (IMG-5)."""

    task_id: str = "IMG-5"
    kind: str = "binary"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """COCO 데이터셋에서 seed 고정 subset을 로드 (이진 분류: 사람 포함 여부).

        각 샘플은 image + objects(카테고리 ID list)를 갖는다.
        objects의 category에 0(person)이 있으면 label=1(사람 있음),
        없으면 label=0(사람 없음).

        COCO 데이터셋은 대부분 사람을 포함하므로 밸런스 확보를 위해
        더 큰 n을 요청한 후 Python에서 n개까지만 사용.
        load_hf_split은 streaming이므로 n을 크게 설정해도 필요한 만큼만 로드.

        load_hf_split은 list[dict]를 반환하므로, 각 행을 dict로 처리.
        """
        registry = load_registry()
        dataset_entry = resolve_dataset_entry(registry, "img_person")

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "val")
        config_name = dataset_entry.get("config")
        # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
        revision = dataset_entry.get("revision")

        # 보다 더 많은 샘플을 로드해서 클래스 밸런스 확보
        # streaming이므로 load_n을 크게 설정해도 필요한 만큼만 스트리밍 로드
        load_n = max(n * 3, 30)
        hf_ds = load_hf_split(hf_id, split, load_n, seed, config_name, revision)

        samples = []
        sample_id = 0
        class_counts = {0: 0, 1: 0}

        for idx, row in enumerate(hf_ds):
            image = row.get("image")
            objects = row.get("objects")

            if image is None:
                continue

            # objects는 dict with "category" key containing list of category IDs
            category_ids = objects.get("category", []) if isinstance(objects, dict) else []
            if not isinstance(category_ids, list):
                category_ids = [category_ids]

            # 사람 판별: COCO person class ID = 0
            has_person = any(
                isinstance(cat_id, int) and cat_id == 0 for cat_id in category_ids
            )

            label_int = 1 if has_person else 0
            class_counts[label_int] += 1

            sample = Sample(
                sample_id=sample_id,
                inputs={"image": image},
                reference=label_int,
                lang="en",
                meta={
                    "dataset": "img_person",
                    "source_idx": idx,
                    "num_objects": len(category_ids) if isinstance(category_ids, list) else 0,
                    "has_person": has_person,
                },
            )
            samples.append(sample)
            sample_id += 1

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """사람 포함 여부 판별 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        명확한 yes/no 질문으로 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "Is there a person visible in this image? Answer exactly yes or no."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "yes"→1, "no"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시에도 동작.
        부정 표현("no", "not") 우선 검사로 "no person" 오분류 방지.
        파싱 불가 시 None 반환.
        """
        if not raw_text or not raw_text.strip():
            return None

        text_lower = raw_text.strip().lower()

        # 부정 표현 먼저 검사 ("no person"은 0이어야 함)
        if "no" in text_lower or "not" in text_lower or "none" in text_lower:
            return 0

        # 긍정 표현 → 1
        if "yes" in text_lower or "person" in text_lower or "people" in text_lower:
            return 1

        # 숫자로도 시도: 1/0
        if re.search(r"\b1\b", text_lower):
            return 1
        if re.search(r"\b0\b", text_lower):
            return 0

        return None

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """IMG-5 사람 포함 이진 분류 채점. **누적기에 위임**한다(단일 경로 — score()/누적기 드리프트 방지).

        파싱 실패(None)는 누적기가 **오답으로 채점**하고 `n_unparsed`로도 보고한다.
        예전엔 여기서 `p is not None`으로 걸러 분모에서 빼, 정답 1개 + 파싱 실패 29개가
        accuracy 1.0으로 나왔다(2026-08-06 지적). 호출 실패(CALL_FAILED)만 제외된다.
        """
        return self.score_via_accumulator(parsed, samples)

    def make_accumulator(self) -> BinaryAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일한 accuracy/f1/confusion/class_balance 반환."""
        return BinaryAccumulator(class_balance_keys=("class_0", "class_1"), include_confusion=True)



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
    """간단한 end-to-end 테스트: 4샘플로 사람 검출 실행."""
    import sys
    from PIL import Image
    import io
    import urllib.request
    from io import BytesIO

    print("=== IMG-5 사람 포함 여부 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "img_person"}}
    task = Img5Task(task_config, registry)

    # 테스트용 실제 이미지 URL 사용 (COCO 데이터셋의 검증 이미지)
    # person=1인 이미지, person=0인 이미지 혼합
    print("Downloading test images from COCO...\n")

    test_urls = [
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000184613.jpg", 1),  # people with umbrella
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000581921.jpg", 1),  # people
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000222564.jpg", 0),  # dogs (no person)
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000171230.jpg", 0),  # outdoor scene
    ]

    mock_samples = []
    for idx, (url, label) in enumerate(test_urls):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data))
            image.load()

            sample = Sample(
                sample_id=idx,
                inputs={"image": image},
                reference=label,
                lang="en",
                meta={"dataset": "img_person", "source_url": url},
            )
            mock_samples.append(sample)
            print(f"  ✓ Sample {idx}: {url.split('/')[-1]} (person={label})")
        except Exception as e:
            print(f"  ✗ Failed to load {url}: {e}")

    if not mock_samples:
        print("\n✗ Failed to load test images!", file=sys.stderr)
        sys.exit(1)

    samples = mock_samples
    print(f"\nLoaded {len(samples)} test samples")

    # 클래스 분포
    class_0_count = sum(1 for s in samples if s.reference == 0)
    class_1_count = sum(1 for s in samples if s.reference == 1)
    print(f"Class balance: class_0={class_0_count}, class_1={class_1_count}\n")

    for sample in samples:
        print(
            f"[Sample {sample.sample_id}] Image: {sample.inputs['image'].size}, "
            f"Label: {sample.reference} (person={'yes' if sample.reference else 'no'})"
        )
    print()

    # 프롬프트 생성 및 모델 호출
    print("Calling FMAPI (databricks-claude-opus-5)...\n")

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
                    max_tokens=64,
                    extra_params={"thinking": {"type": "disabled"}},
                )

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"  Raw response: {response.text[:80]}")
                print(f"  Parsed: {parsed}")
                print(f"  Expected: {sample.reference}")
                print()

            # 채점
            print("\n=== Scoring (binary_metrics) ===\n")
            scores = task.score(parsed_outputs, samples)

            print(f"Accuracy: {scores['accuracy']:.3f}")
            print(f"F1: {scores['f1']:.3f}")
            print(f"Confusion matrix: {scores['confusion_matrix']}")
            print(f"Class balance: {scores['class_balance']}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
