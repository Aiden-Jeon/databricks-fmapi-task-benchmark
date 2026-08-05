"""IMG-2 어휘 정규화 테스트 (이슈 D).

gold가 COCO 80클래스인데 모델은 자유 명사로 답해 micro_f1이 0.2에 눌려 있었다
(`umbrellas`≠`umbrella`, `ocean`·`sky`는 어휘 밖). "태그 추출 능력"이 아니라
"COCO 어휘 맞히기"를 재던 문제다. 수정은 두 겹:
1) 프롬프트에 닫힌 라벨 집합 제공 (본 수정)
2) 파서에서 복수형·불규칙 변형 정규화 + 어휘 밖 제거 (방어선)

실측 효과: 옛 응답에 파서만 적용해도 opus 0.202→0.737, sol 0.248→0.757.
"""

from src.tasks.img_2 import COCO_LABELS, _COCO_ALIASES, _normalize_coco_tag


def test_exact_label_passes():
    assert _normalize_coco_tag("umbrella") == "umbrella"
    assert _normalize_coco_tag("dining table") == "dining table"


def test_case_and_whitespace_normalized():
    assert _normalize_coco_tag("  Potted   Plants ") == "potted plant"
    assert _normalize_coco_tag("PERSON") == "person"


def test_regular_plural_stripped():
    """이 버그의 대표 사례 — umbrellas가 오답으로 집계됐다."""
    assert _normalize_coco_tag("umbrellas") == "umbrella"
    assert _normalize_coco_tag("chairs") == "chair"


def test_multiword_plural_stripped():
    """마지막 단어에만 복수형이 붙는 형태."""
    assert _normalize_coco_tag("traffic lights") == "traffic light"
    assert _normalize_coco_tag("wine glasses") == "wine glass"


def test_irregular_forms_via_alias():
    assert _normalize_coco_tag("people") == "person"
    assert _normalize_coco_tag("knives") == "knife"
    assert _normalize_coco_tag("tvs") == "tv"
    assert _normalize_coco_tag("motorbike") == "motorcycle"


def test_out_of_vocabulary_dropped():
    """어휘 밖 단어는 버린다 — gold에 없어 precision만 깎는 잡음이다."""
    for t in ("ocean", "sky", "cruise ships", "pier", "water"):
        assert _normalize_coco_tag(t) is None


def test_empty_and_none_safe():
    assert _normalize_coco_tag("") is None
    assert _normalize_coco_tag(None) is None
    assert _normalize_coco_tag("   ") is None


def test_all_alias_targets_are_real_labels():
    """alias 타깃이 전부 실제 라벨이어야 한다.

    COCO_CATEGORY_MAP은 일부 id가 빠진 78개짜리라, 표에만 있는 라벨
    (예: 'cell phone')로 매핑하면 그 태그는 영원히 오답이 된다.
    """
    invalid = {v for v in _COCO_ALIASES.values() if v not in COCO_LABELS}
    assert not invalid, f"라벨 집합에 없는 alias 타깃: {sorted(invalid)}"


def test_prompt_provides_closed_vocabulary():
    """프롬프트가 라벨 집합을 실제로 포함하는지 — 이슈 D의 본 수정."""
    from src.tasks.base import Sample
    from src.tasks.loader import discover_tasks
    from src.datasets_loader import load_registry
    from PIL import Image

    inst = discover_tasks()["IMG-2"]({}, load_registry())
    sample = Sample(
        sample_id=0,
        inputs={"image": Image.new("RGB", (8, 8))},
        reference={"person"},
        lang="en",
    )
    text = str(inst.build_prompt(sample))
    assert "Label set:" in text
    assert "umbrella" in text and "person" in text
    assert "Do not invent labels" in text
