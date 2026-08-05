"""태스크 플러그인 베이스 인터페이스 (Phase 1).

각 태스크(TXT-4, TXT-5, ...)는 이 Task를 상속해 4가지를 구현한다:
- load_samples(): 데이터셋에서 Sample 리스트 생성 (seed 고정 subset)
- build_prompt(sample): 모델에 보낼 프롬프트/메시지 구성
- parse_output(raw_text): 모델 평문 응답 → 채점 가능한 형태로 파싱
- score(parsed_list, samples): 파싱 결과 전체 → 집계 메트릭 dict

runner는 task_id로 플러그인을 동적 로드해 (모델 × reasoning_mode)로 순회한다.
채점은 per-sample이 아니라 태스크 단위 집계(정확도·F1 등)이므로 score는 전체를 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """태스크의 개별 평가 샘플."""

    sample_id: int
    inputs: dict[str, Any]      # 프롬프트 구성에 필요한 입력 (text, image_url, question 등)
    reference: Any              # 정답 (str / list / dict / int — 태스크마다 다름)
    lang: str = "en"            # ko | en (채점 시 토크나이저 선택)
    meta: dict[str, Any] = field(default_factory=dict)  # 갤러리 표시용 부가정보


class Task:
    """모든 태스크 플러그인의 베이스.

    서브클래스는 클래스 속성 `task_id`, `kind`와 아래 메서드를 구현한다.
    is_vision=True면 이미지 입력 태스크(vision 미지원 모델은 runner가 N/A 스킵).
    """

    task_id: str = ""
    kind: str = ""              # generation | qa | classification | binary | multilabel | extraction
    is_vision: bool = False
    sensitive: bool = False     # True면 리포트 갤러리에서 입력 이미지 숨김 (IMG-4 NSFW, D3)

    def __init__(self, config: dict[str, Any], registry: dict[str, Any]) -> None:
        """config: tasks.yaml의 해당 태스크 항목. registry: datasets/registry.yaml."""
        self.config = config
        self.registry = registry

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """데이터셋에서 seed 고정 subset n개를 Sample로 반환."""
        raise NotImplementedError

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """FMAPI messages 리스트를 반환 (adapters.fmapi.build_text_message 등 활용)."""
        raise NotImplementedError

    def parse_output(self, raw_text: str, sample: Sample) -> Any:
        """모델 평문 응답을 채점 가능한 형태로 파싱 (분류=라벨, 생성=텍스트 등)."""
        raise NotImplementedError

    def score(self, parsed: list[Any], samples: list[Sample]) -> dict[str, Any]:
        """파싱 결과 전체를 집계해 메트릭 dict 반환. parsed[i]는 samples[i]에 대응."""
        raise NotImplementedError

    def score_via_accumulator(self, parsed: list[Any], samples: list[Sample]) -> dict[str, Any]:
        """`make_accumulator()`에 그대로 흘려 넣어 채점한다 — score()의 표준 구현.

        **두 경로가 갈리지 않게 하는 장치다.** 예전엔 태스크마다 score()가 따로 구현돼,
        누적기는 파싱 실패(None)를 오답으로 채점하는데 score()는 `p is not None` 필터로
        분모에서 빼는 불일치가 생겼다(실측: 정답 1 + 파싱실패 29 → accuracy 1.0).
        누적기를 단일 진실로 삼으면 그 종류의 드리프트가 구조적으로 불가능해진다.

        누적기가 없는 태스크는 이 메서드를 쓰지 않고 score()를 직접 구현한다.
        """
        acc = self.make_accumulator()   # type: ignore[attr-defined]
        for p, s in zip(parsed, samples):
            acc.add(p, s)
        return acc.finalize()


# 태스크 레지스트리: task_id → Task 서브클래스. 각 태스크 모듈이 import 시 register()로 등록.
_REGISTRY: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    """Task 서브클래스를 task_id로 등록하는 데코레이터."""
    if not cls.task_id:
        raise ValueError(f"{cls.__name__}에 task_id가 없음")
    _REGISTRY[cls.task_id] = cls
    return cls


def get_task_class(task_id: str) -> type[Task] | None:
    """등록된 태스크 클래스를 반환. 없으면 None (미구현 태스크)."""
    return _REGISTRY.get(task_id)


def all_registered() -> dict[str, type[Task]]:
    return dict(_REGISTRY)
