"""IMG-2 이미지 태그 추출 태스크 (multilabel).

이 태스크는 이미지에서 주요 객체(object)를 태그로 추출하도록 한다.
- 데이터셋: detection-datasets/coco (COCO objects.category config)
- 메트릭: precision, recall, f1 (multilabel_prf)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry, get_label_names
from src.scoring.metrics import multilabel_prf
from src.tasks.base import Task, Sample, register


# COCO 80개 클래스 맵 (id → name)
COCO_CATEGORY_MAP = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow",
    21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack",
    26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee",
    31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat",
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket",
    40: "bottle", 41: "wine glass", 42: "cup", 43: "fork", 44: "knife",
    45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich",
    50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza",
    55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant",
    60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 64: "laptop",
    65: "mouse", 66: "remote", 67: "keyboard", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


@register
class Img2Task(Task):
    """이미지 태그 추출 multilabel 태스크 (IMG-2)."""

    task_id: str = "IMG-2"
    kind: str = "multilabel"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """COCO 객체 검출 데이터셋에서 이미지+태그를 로드.

        detection-datasets/coco의 objects.category에서 각 이미지의
        객체 카테고리를 수집해 gold tag set을 구성한다.

        load_hf_split은 list[dict]를 반환하므로, 각 행을 dict로 처리.
        """
        registry = load_registry()
        dataset_entry = resolve_dataset_entry(registry, "img_tags")

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "val")
        config_name = dataset_entry.get("config")

        # 라벨 이름 얻기 (streaming 메타만 조회, 데이터 다운로드 없음)
        label_names = get_label_names(hf_id, split, config_name, "objects.category")

        # 라벨 이름이 없으면 hardcoded COCO-80 맵 사용
        if label_names:
            # category ID → name 맵 (index가 ID임)
            category_map = {i: name for i, name in enumerate(label_names)}
        else:
            category_map = COCO_CATEGORY_MAP

        # HF 데이터셋 로드 (list[dict] 반환)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name)

        samples = []
        sample_id = 0

        for idx, row in enumerate(hf_ds):
            image = row.get("image")
            objects = row.get("objects")

            if image is None or objects is None:
                continue

            # objects는 dict with "category" key containing list of category IDs
            category_ids = objects.get("category", []) if isinstance(objects, dict) else []
            if not isinstance(category_ids, list):
                category_ids = [category_ids]

            # Category ID → name 맵핑
            tags = set()
            for cat_id in category_ids:
                try:
                    cat_id_int = int(cat_id)
                    if cat_id_int in category_map:
                        tags.add(category_map[cat_id_int])
                except (ValueError, TypeError):
                    pass

            if not tags:
                continue

            sample = Sample(
                sample_id=sample_id,
                inputs={"image": image},
                reference=tags,  # set of tag strings
                lang="en",
                meta={
                    "dataset": "img_tags",
                    "source_idx": idx,
                },
            )
            samples.append(sample)
            sample_id += 1

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """이미지에서 주요 객체를 태그로 추출하는 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "List the main objects visible in this image as a comma-separated list of simple nouns."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> set[str]:
        """모델 응답을 정규화된 태그 set으로 파싱.

        쉼표/줄바꿈으로 분할. 단어가 아닌 문장 부분은 제외.
        짧고 명사 같은 항목만 선택 (한두 단어).
        """
        if not raw_text:
            return set()

        import re

        # 먼저 첫 줄에서 구분자 찾기 (많은 모델이 "Looking at ... :" 식으로 시작)
        lines = raw_text.split("\n")
        content_start_idx = 0
        for i, line in enumerate(lines):
            # ":" 이후의 라인부터 실제 항목 시작
            if ":" in line:
                content_start_idx = i
                break

        # 콘텐츠 부분만 처리
        text_to_parse = "\n".join(lines[content_start_idx:])

        # 쉼표로 먼저 분할
        tags = []
        for line in text_to_parse.split("\n"):
            for item in line.split(","):
                tag = item.strip().lower()

                # 빈 항목, 문장 부분 필터링
                if not tag or len(tag) < 2:
                    continue

                # 너무 긴 문장 필터 (3단어 이상이면 보통 설명문)
                words = tag.split()
                if len(words) > 3:
                    continue

                # 기호 제거 (마침표, 물음표 등)
                tag = re.sub(r"[.!?;:()]", "", tag).strip()
                if tag:
                    tags.append(tag)

        return set(tags)

    def score(self, parsed: list[set[str]], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 태그 set을 multilabel_prf로 평가.

        각 샘플의 예측 태그 set과 gold 태그 set을 비교.
        """
        if not parsed or not samples:
            return {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "n_evaluated": 0,
            }

        # 유효한 샘플만 필터링
        valid_indices = [
            i for i, (pred, sample) in enumerate(zip(parsed, samples))
            if pred is not None and sample.reference
        ]

        if not valid_indices:
            return {
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "n_evaluated": 0,
            }

        pred_sets = [parsed[i] for i in valid_indices]
        gold_sets = [samples[i].reference for i in valid_indices]

        # multilabel_prf 호출
        metrics = multilabel_prf(pred_sets, gold_sets)

        metrics["n_evaluated"] = len(valid_indices)
        return metrics


if __name__ == "__main__":
    """간단한 end-to-end 테스트: 2샘플로 태그 추출 실행."""
    import sys
    from PIL import Image
    import io

    print("=== IMG-2 태그 추출 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "img_tags"}}
    task = Img2Task(task_config, registry)

    # 테스트용 실제 이미지 URL 사용 (IMG-1과 동일한 COCO-Karpathy 데이터 활용)
    print("Downloading test images from COCO-Karpathy...\n")

    import urllib.request
    from io import BytesIO

    # COCO-Karpathy validation 샘플 이미지
    test_urls = [
        "http://images.cocodataset.org/val2014/COCO_val2014_000000184613.jpg",  # yak with umbrella
        "http://images.cocodataset.org/val2014/COCO_val2014_000000393220.jpg",  # train
    ]

    mock_samples = []
    for idx, url in enumerate(test_urls):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data))
            image.load()

            # 간단한 mock 태그 (실제 이미지 내용에 맞는 태그)
            mock_tags = [
                {"person", "umbrella", "animal", "outdoor"},
                {"train", "railway", "vehicle", "station"},
            ]

            sample = Sample(
                sample_id=idx,
                inputs={"image": image},
                reference=mock_tags[idx],
                lang="en",
                meta={"dataset": "img_tags", "source_idx": idx, "url": url},
            )
            mock_samples.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to load {url}: {e}")

    samples = mock_samples
    print(f"Loaded {len(samples)} test images\n")

    if not samples:
        print("Failed to load test images. Using dummy samples...\n")
        def create_dummy_image(size=(256, 256), color=(100, 150, 200)):
            img = Image.new("RGB", size, color)
            return img

        samples = [
            Sample(
                sample_id=0,
                inputs={"image": create_dummy_image(color=(100, 150, 200))},
                reference={"person", "chair", "desk", "laptop"},
                lang="en",
                meta={"dataset": "img_tags", "source_idx": 0},
            ),
        ]

    for sample in samples[:1]:
        print(f"[Sample {sample.sample_id}]")
        print(f"  Image: PIL shape {sample.inputs['image'].size}")
        print(f"  Gold tags: {sample.reference}")
        print()

    # 프롬프트 생성 및 모델 호출
    print("Calling FMAPI (claude-opus-5)...\n")

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
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
                print(f"  Parsed tags: {parsed}")
                print(f"  Gold tags: {sample.reference}")
                print()

            # 채점
            print("\n=== Scoring (multilabel_prf) ===\n")
            scores = task.score(parsed_outputs, samples)

            print(f"Micro Precision: {scores['micro_precision']:.3f}")
            print(f"Micro Recall:    {scores['micro_recall']:.3f}")
            print(f"Micro F1:        {scores['micro_f1']:.3f}")
            print(f"Macro Precision: {scores['macro_precision']:.3f}")
            print(f"Macro Recall:    {scores['macro_recall']:.3f}")
            print(f"Macro F1:        {scores['macro_f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
