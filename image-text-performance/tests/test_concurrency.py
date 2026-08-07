"""제한 병렬 map 헬퍼 계약 테스트 — 순서 보존·예외 비삼킴·순차 동치.

이 헬퍼는 FMAPI 호출을 겹쳐 실행해 속도를 올리는 용도지만, **채점 수치에는 전혀 영향을
주지 않아야** 한다. 그 안전성의 근거가 아래 세 성질이다(모듈 docstring의 계약).
"""

from __future__ import annotations

import threading
import time

import pytest

from src.adapters.concurrency import map_concurrent


def test_preserves_input_order_regardless_of_completion_order():
    """완료가 뒤섞여도 결과는 입력 순서 그대로 — 채점이 순서에 의존해도 안전하다."""
    # 뒤 항목일수록 빨리 끝나게 해 완료 순서를 입력 순서와 반대로 만든다.
    def fn(i: int) -> int:
        time.sleep((10 - i) * 0.005)
        return i * 10

    out = map_concurrent(fn, list(range(10)), max_workers=8)
    values = [v for v, _ in out]
    assert values == [i * 10 for i in range(10)], f"입력 순서가 깨졌다: {values}"
    assert all(e is None for _, e in out)


def test_captures_exceptions_without_raising():
    """예외를 삼키지 않고 (None, 예외)로 돌려준다 — 호출부가 실패 종류를 판정한다."""
    def fn(i: int) -> int:
        if i % 2 == 0:
            raise ValueError(f"boom-{i}")
        return i

    out = map_concurrent(fn, list(range(4)), max_workers=4)
    # 짝수는 예외, 홀수는 값 — 위치가 보존된다.
    assert out[0][0] is None and isinstance(out[0][1], ValueError)
    assert out[1] == (1, None)
    assert out[2][0] is None and isinstance(out[2][1], ValueError)
    assert out[3] == (3, None)
    assert "boom-0" in str(out[0][1])


def test_single_worker_runs_sequentially_in_order():
    """max_workers<=1이면 스레드 없이 순차 실행(순차 경로와 동일 동작)."""
    seen: list[int] = []

    def fn(i: int) -> int:
        seen.append(i)   # 순차이므로 GIL 경합 없이 제출 순서대로 쌓인다
        return i

    out = map_concurrent(fn, [3, 1, 2], max_workers=1)
    assert [v for v, _ in out] == [3, 1, 2]
    assert seen == [3, 1, 2], "단일 워커는 제출 순서대로 실행해야 한다"


def test_empty_input_returns_empty():
    assert map_concurrent(lambda x: x, [], max_workers=8) == []


def test_actually_runs_in_parallel():
    """겹쳐 실행하는지 확인 — 8개를 8워커로 돌리면 순차 합보다 훨씬 빠르다."""
    def slow(i: int) -> int:
        time.sleep(0.05)
        return i

    t0 = time.perf_counter()
    out = map_concurrent(slow, list(range(8)), max_workers=8)
    elapsed = time.perf_counter() - t0

    assert [v for v, _ in out] == list(range(8))
    # 순차면 8*0.05=0.4s. 병렬이면 ~0.05s. 여유 있게 0.2s 미만으로 본다.
    assert elapsed < 0.2, f"병렬 실행이 아니다(경과 {elapsed:.3f}s)"


def test_workers_bounded_by_item_count():
    """동시 실행 스레드 수가 항목 수(=워커 상한)를 넘지 않는다."""
    active = {"cur": 0, "peak": 0}
    lock = threading.Lock()

    def fn(i: int) -> int:
        with lock:
            active["cur"] += 1
            active["peak"] = max(active["peak"], active["cur"])
        time.sleep(0.02)
        with lock:
            active["cur"] -= 1
        return i

    map_concurrent(fn, list(range(3)), max_workers=8)
    assert active["peak"] <= 3, f"항목 3개인데 동시 실행 {active['peak']} — 워커 상한 초과"
