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
from src.scoring.accumulators import MultilabelAccumulator
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


COCO_LABELS = frozenset(COCO_CATEGORY_MAP.values())

# 복수형 규칙으로 안 잡히는 실측 변형만 최소한으로 매핑한다. 동의어를 넓게 받으면
# "COCO 어휘를 아는지"가 아니라 매핑 테이블 품질을 재게 되므로, 불규칙 복수와
# 명백한 동일 지시어에 한정한다.
_COCO_ALIASES = {
    "people": "person", "men": "person", "women": "person", "man": "person",
    "woman": "person", "child": "person", "children": "person", "boy": "person",
    "girl": "person", "kid": "person", "kids": "person",
    "knives": "knife", "sheeps": "sheep", "buses": "bus", "sandwiches": "sandwich",
    "benches": "bench", "glasses": "wine glass", "couches": "couch",
    "television": "tv", "televisions": "tv", "tvs": "tv", "monitor": "tv",
    "motorbike": "motorcycle",
    "plane": "airplane", "aeroplane": "airplane", "airplanes": "airplane",
    "table": "dining table", "tables": "dining table", "plant": "potted plant",
    "plants": "potted plant", "flower vase": "vase", "bicycles": "bicycle",
    "bike": "bicycle", "bikes": "bicycle", "hairdryer": "hair drier",
    "hair dryer": "hair drier", "sports balls": "sports ball", "ball": "sports ball",
}


def _normalize_coco_tag(tag: str) -> str | None:
    """모델이 낸 태그를 COCO 라벨로 정규화. 어휘 밖이면 None.

    순서: 소문자·공백 정리 → 그대로 일치 → alias 표 → 규칙적 복수형(-s/-es/-ies) 제거.
    gold가 COCO 80클래스라 어휘 밖 단어(`ocean`, `sky`)는 어차피 오답이므로 버린다.
    """
    t = " ".join((tag or "").lower().split())
    if not t:
        return None
    if t in COCO_LABELS:
        return t
    # alias 타깃도 라벨 집합에 있는지 확인한다 — COCO_CATEGORY_MAP은 일부 id가 빠진
    # 78개짜리라, 표에만 있고 실제로는 없는 라벨(예: cell phone)을 내보내면 영원히 오답이 된다.
    alias = _COCO_ALIASES.get(t)
    if alias and alias in COCO_LABELS:
        return alias
    # 규칙적 복수형: umbrellas→umbrella, buses→bus, berries→berry
    for cand in (
        t[:-1] if t.endswith("s") else None,
        t[:-2] if t.endswith("es") else None,
        (t[:-3] + "y") if t.endswith("ies") else None,
    ):
        if cand and cand in COCO_LABELS:
            return cand
    # 복수형이 여러 단어의 마지막에 붙은 경우(traffic lights → traffic light)
    parts = t.split()
    if len(parts) > 1 and parts[-1].endswith("s"):
        cand = " ".join(parts[:-1] + [parts[-1][:-1]])
        if cand in COCO_LABELS:
            return cand
    return None


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
        # revision을 넘겨 데이터를 그 시점으로 고정한다(registry의 revision 필드). 없으면 None.
        revision = dataset_entry.get("revision")

        # 라벨 이름 얻기 (streaming 메타만 조회, 데이터 다운로드 없음)
        label_names = get_label_names(hf_id, split, config_name, "objects.category")

        # 라벨 이름이 없으면 hardcoded COCO-80 맵 사용
        if label_names:
            # category ID → name 맵 (index가 ID임)
            category_map = {i: name for i, name in enumerate(label_names)}
        else:
            category_map = COCO_CATEGORY_MAP

        # HF 데이터셋 로드 (list[dict] 반환)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name, revision)

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

        **닫힌 어휘(COCO 80클래스)를 프롬프트에 제공한다 (2026-08-05 수정).**
        이전에는 "simple nouns"만 요구해서 모델이 자유 명사로 답했고(`cruise ships,
        ocean, sky, pier, umbrellas`), 정답은 COCO 80클래스라 단복수 차이(`umbrellas`
        ≠ `umbrella`)나 어휘 밖 단어(`ocean`, `sky`)가 전부 오답으로 집계돼 micro_f1이
        0.2에 눌렸다. 그건 "태그 추출 능력"이 아니라 "COCO 어휘 맞히기"를 잰 것이다.
        멀티라벨 분류의 표준 관례대로 후보 라벨을 주고 그 안에서만 고르게 한다.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        labels = ", ".join(sorted(COCO_CATEGORY_MAP.values()))
        prompt = (
            "List every object from the label set below that is visible in this image.\n\n"
            f"Label set: {labels}\n\n"
            "Rules: output ONLY a comma-separated list of labels copied exactly from the "
            "label set above (same spelling, singular form). Do not invent labels that are "
            "not in the set. No explanation, no counts, no other text."
        )
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> set[str] | None:
        """모델 응답을 정규화된 태그 set으로 파싱. **호출 실패는 None**(채점 제외).

        쉼표/줄바꿈으로 분할. 단어가 아닌 문장 부분은 제외.
        짧고 명사 같은 항목만 선택 (한두 단어).

        **호출 실패 처리 (2026-08-05 수정)**: 러너는 호출이 실패하면 `model_output`에
        `"__ERROR__: ..."`를 넣는다. 예전엔 그 문자열을 정상 텍스트로 파싱해 빈 set이
        되고 그게 **0점으로 채점**됐다(실측: opus IMG-2가 502로 11/30 실패 → micro_f1
        0.671, 성공 19건만 보면 0.786). 분류 태스크들(IMG-3/4/5)은 None을 돌려 이미
        제외하고 있었는데 이 태스크만 0점으로 섞었다. `score()`·누적기가 None을 제외하므로
        여기서 None을 돌려주면 실패가 성능으로 오독되지 않는다.
        """
        if not raw_text:
            return None
        if str(raw_text).startswith("__ERROR__"):
            return None

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
                # 목록 기호·번호 제거("- person", "1. person")
                tag = re.sub(r"^([-*•]|\d+[.)])\s*", "", tag).strip()
                if tag:
                    tags.append(tag)

        # 어휘 정규화: 프롬프트로 COCO 라벨을 요구하지만 모델이 복수형·동의어로 답할 수
        # 있어(실측: `umbrellas`, `tvs`, `people`) 라벨 집합에 맞춰 붙인다. 어휘 밖 단어는
        # 버린다 — gold가 COCO 80클래스뿐이라 남겨두면 precision만 깎는 잡음이다.
        return {norm for norm in (_normalize_coco_tag(t) for t in tags) if norm}

    def score(self, parsed: list[set[str]], samples: list[Sample]) -> dict[str, Any]:
        """IMG-2 태그 추출(멀티라벨). **누적기에 위임**한다(단일 경로 — score()/누적기 드리프트 방지).

        파싱 실패(None)는 빈 태그셋으로 채점(precision/recall 0)되고 호출 실패(CALL_FAILED)만 채점에서 제외된다.
        예전엔 이 경로가 별도로 구현돼 파싱 실패를 분모에서 빼거나(점수 부풀림) 예외를
        던졌다(셀 전체가 error로 죽어 정상 샘플까지 버려짐) — 2026-08-06 지적.
        """
        return self.score_via_accumulator(parsed, samples)

    def make_accumulator(self) -> MultilabelAccumulator:
        """스트리밍 O(1) 채점기. score()와 동일(micro/macro PRF + n_evaluated).

        파싱 실패(None)는 **빈 태그셋으로 채점**한다(precision/recall 0) — 태그를 못 낸 것은
        실제 능력 문제다. 예전엔 분모에서 빼서, 형식을 못 맞추는 모델이 성공분만으로 높은
        F1을 받았다(2026-08-06 지적). 호출 실패는 CALL_FAILED sentinel로 따로 제외된다.
        """
        return MultilabelAccumulator(
            valid_fn=lambda p, s: bool(s.reference),
            normalize_fn=lambda p: p if p is not None else set(),
        )



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
        # 합성/더미 폴백 없음 — 실제 데이터가 안 되면 정직하게 실패한다.
        print("실제 테스트 이미지를 로드하지 못했습니다. 네트워크·데이터셋을 확인하세요.")
        raise SystemExit(1)

    for sample in samples[:1]:
        print(f"[Sample {sample.sample_id}]")
        print(f"  Image: PIL shape {sample.inputs['image'].size}")
        print(f"  Gold tags: {sample.reference}")
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
