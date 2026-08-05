"""TXT-2 답 추출 테스트 (이슈 L).

TXT-2는 정답이 짧은 셀값(`['4']`, `['George Burley']`)인데 모델이 근거를 곁들인
산문으로 답해 accuracy가 0.0~0.1로 눌려 있었다(judge는 같은 셀을 3.7~4.4로 평가 —
표를 못 읽은 게 아니라 형식이 안 맞았다는 뜻). 근본 해결은 프롬프트에서 짧은 답을
요구하는 것이고, 이 파서는 그래도 설명을 붙이는 모델에 대한 방어선이다.

케이스는 `2026-08-05T07-09` run의 실제 opus/sol 출력에서 가져왔다.
"""

from src.tasks.base import Sample
from src.tasks.loader import discover_tasks
from src.datasets_loader import load_registry


def _task():
    return discover_tasks()["TXT-2"]({}, load_registry())


def _sample(reference):
    return Sample(
        sample_id=0,
        inputs={"table": "| a |\n|---|\n| 1 |", "question": "q?"},
        reference=reference,
        lang="en",
    )


def _parse(raw, reference=("4",)):
    return _task().parse_output(raw, _sample(list(reference)))


def test_bare_answer_passes_through():
    """짧은 답은 그대로(프롬프트가 유도하는 정상 경로)."""
    assert _parse("4") == "4"
    assert _parse("The Sound Of Trees") == "The Sound Of Trees"


def test_strips_answer_label():
    """'Answer:' 라벨 뒤만 남긴다."""
    assert _parse("Answer: 4") == "4"
    assert _parse("Final answer: George Burley") == "George Burley"


def test_takes_last_line_when_reasoning_precedes():
    """설명이 앞에 오고 결론이 마지막 줄인 형태."""
    raw = "Looking at the table, I count the rows.\nThere are several.\n36"
    assert _parse(raw) == "36"


def test_strips_markdown_emphasis_including_unpaired():
    """`**`가 짝이 안 맞게 잘려 오는 실측 형태(**Answer: 2 matches** were played)."""
    assert _parse("**Answer: 2 matches** were played in May 2010") == "2 matches were played in May 2010"
    assert _parse("**119**") == "119"


def test_strips_trailing_period_and_backticks():
    assert _parse("1997.") == "1997"
    assert _parse("`Clint Bolton`") == "Clint Bolton"


def test_keeps_list_when_last_line_is_an_item():
    """마지막 줄이 목록 항목이면 결론이 아니라 열거 → 통째로 유지(잘못 잘라내지 않는다)."""
    raw = "The parishes are:\n1. St Mary\n2. Our Lady"
    assert _parse(raw) == raw.strip()


def test_empty_and_none_are_safe():
    assert _parse("") == ""
    assert _task().parse_output(None, _sample(["4"])) == ""


def test_multi_value_answer_preserved():
    """다중 값 답은 구분자를 유지한다(프롬프트가 ', ' 구분을 요구)."""
    assert _parse("Morocco, France, Spain") == "Morocco, France, Spain"


def test_prompt_requires_short_answer():
    """프롬프트가 '설명 없이 답만'을 실제로 요구하는지 — 이게 이슈 L의 본 수정이다."""
    msgs = _task().build_prompt(_sample(["4"]))
    text = str(msgs)
    assert "ONLY the answer" in text
    assert "no explanation" in text
