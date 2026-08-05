"""전체 태스크 등록·인터페이스 정합 검증 (호출 없음, 빠름).

config/tasks.yaml의 모든 태스크(현재 14개)가 등록되고 task_id가 일치하며,
필수 메서드가 구현됐는지 확인. 실제 FMAPI 호출은 하지 않는다(비용·시간 절약).
"""

import sys

import yaml

from src.tasks.base import Task
from src.tasks.loader import discover_tasks


def main() -> int:
    registered = discover_tasks()
    tasks_cfg = yaml.safe_load(open("config/tasks.yaml", encoding="utf-8"))
    expected = [t["id"] for t in tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])]

    print(f"config 태스크 {len(expected)}개: {expected}")
    print(f"등록된 태스크 {len(registered)}개: {sorted(registered)}")
    print()

    missing = [t for t in expected if t not in registered]
    extra = [t for t in registered if t not in expected]

    ok = True
    for tid in expected:
        cls = registered.get(tid)
        if cls is None:
            print(f"  ❌ {tid}: 미등록")
            ok = False
            continue
        # 필수 메서드가 base가 아니라 서브클래스에서 오버라이드됐는지
        overridden = all(
            getattr(cls, m) is not getattr(Task, m)
            for m in ("load_samples", "build_prompt", "parse_output", "score")
        )
        vision = "vision" if getattr(cls, "is_vision", False) else "text "
        sens = " [sensitive]" if getattr(cls, "sensitive", False) else ""
        print(f"  {'✅' if overridden else '⚠️ '} {tid} ({vision}) {cls.__name__}{sens}")
        if not overridden:
            ok = False

    if missing:
        print(f"\n  누락: {missing}")
        ok = False
    if extra:
        print(f"\n  config에 없는 등록: {extra}")

    print("\n" + ("전체 태스크 정합 OK" if ok and not missing else "정합 문제 있음"))
    return 0 if ok and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
