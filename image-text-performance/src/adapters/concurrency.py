"""제한 병렬 map 유틸 — FMAPI 호출(네트워크 대기)을 겹쳐 실행해 벽시계 시간을 줄인다.

러너는 (모델×태스크×모드) 한 셀 안에서 샘플마다 `fmapi.chat()`을 **순차** 호출해 왔다.
호출 하나가 수 초(glm은 6~7초, judge는 2.8초+)라, 셀 하나가 샘플 수 × 지연으로 길어지고
전체 실행이 1.5~2시간이 된다. rate limit은 넉넉(200k req/500M tok, 429 미관측)하므로
호출을 병렬로 겹치면 대기 시간이 곧바로 줄어든다.

이 헬퍼가 지키는 계약(러너의 기존 불변식을 깨지 않기 위한 핵심):
1. **입력 순서 보존.** 결과를 입력 순서 그대로 돌려준다. 누적기는 교환법칙이 성립하지만,
   러너의 SampleResult 기록·갤러리 버퍼·judge 점수 리스트는 순서에 의존한다. 완료 순서가
   아니라 제출 순서로 정렬해 돌려줘, 동시성이 채점 수치에 전혀 영향을 주지 않게 한다.
2. **예외를 삼키지 않는다.** 각 항목의 결과를 `(value, exception)`로 돌려준다. 러너는
   호출 실패를 `CALL_FAILED`로, 파싱 실패를 `None`으로 **다르게** 분류하는데(이 벤치마크의
   반복된 사고 원인이 이 둘을 섞은 것이었다), 그 판정은 호출부가 해야 한다. 헬퍼가 예외를
   잡아 로그만 남기면 그 축이 무너진다.
3. **워커 1이면 순차와 동일.** concurrency<=1이면 스레드를 만들지 않고 그 자리에서 실행한다
   (테스트·디버깅·병렬 비활성화 경로가 순차 코드와 bit-identical하게 동작).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(
    fn: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int,
) -> list[tuple[R | None, Exception | None]]:
    """`fn`을 `items`에 병렬 적용하고, **입력 순서대로** `(결과, 예외)` 리스트를 돌려준다.

    각 항목에 대해 정확히 하나가 채워진다: 성공이면 `(값, None)`, 실패면 `(None, 예외)`.
    호출부가 예외를 보고 실패 종류를 판정한다(삼키지 않는다 — 모듈 docstring 계약 2).

    max_workers<=1이면 스레드 없이 순차 실행한다(계약 3). 빈 입력이면 빈 리스트.
    """
    n = len(items)
    if n == 0:
        return []

    # 워커가 항목 수보다 많을 이유가 없다. 1 이하이면 순차 경로.
    workers = min(max_workers, n)
    if workers <= 1:
        out: list[tuple[R | None, Exception | None]] = []
        for it in items:
            try:
                out.append((fn(it), None))
            except Exception as e:  # noqa: BLE001 — 호출부가 종류를 판정한다
                out.append((None, e))
        return out

    results: list[tuple[R | None, Exception | None]] = [(None, None)] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # 인덱스를 함께 넘겨 완료 순서와 무관하게 제출 위치에 결과를 꽂는다(순서 보존).
        futures = {ex.submit(_guarded, fn, it): i for i, it in enumerate(items)}
        for fut, i in futures.items():
            # _guarded가 예외를 잡아 튜플로 돌려주므로 fut.result()는 던지지 않는다.
            results[i] = fut.result()
    return results


def _guarded(fn: Callable[[T], R], item: T) -> tuple[R | None, Exception | None]:
    """`fn(item)`을 실행해 `(결과, 예외)`로 감싼다. 워커 스레드 안에서 실행된다."""
    try:
        return (fn(item), None)
    except Exception as e:  # noqa: BLE001 — 호출부가 CALL_FAILED/None으로 분류한다
        return (None, e)
