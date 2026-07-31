"""전체 태스크의 실제 데이터 로딩 검증 (합성 우회 색출).

각 태스크가 2샘플을 실제로 로드하는지, 합성/더미로 빠지지 않는지 확인.
FMAPI 호출은 안 함(로딩만). script-based 실패·합성 fallback을 드러낸다.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from src.datasets_loader import load_registry
from src.tasks.loader import discover_tasks
import yaml


def main() -> int:
    registered = discover_tasks()
    registry = load_registry()
    tasks_cfg = yaml.safe_load(open("config/tasks.yaml", encoding="utf-8"))
    cfgs = {t["id"]: t for t in tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])}

    order = ["IMG-1", "IMG-2", "IMG-3", "IMG-4", "IMG-5",
             "TXT-1", "TXT-2", "TXT-3", "TXT-4", "TXT-5", "TXT-6", "TXT-7", "TXT-8"]
    problems = []
    for tid in order:
        cls = registered.get(tid)
        inst = cls(cfgs.get(tid, {}), registry)
        try:
            samples = inst.load_samples(2, 42)
            n = len(samples)
            # 합성 흔적 탐지: meta에 synthetic/mock/test 표시
            synth = any(
                any(k in str(s.meta).lower() for k in ("synthetic", "mock", "dummy", "합성"))
                for s in samples
            )
            ref_preview = str(samples[0].reference)[:50] if samples else "—"
            flag = "🔴 합성흔적" if synth else ("✅" if n > 0 else "⚠️ 0샘플")
            print(f"  {flag} {tid}: {n}샘플 | ref={ref_preview}")
            if synth or n == 0:
                problems.append(tid)
        except Exception as e:
            print(f"  ❌ {tid}: 로드실패 {type(e).__name__}: {str(e)[:80]}")
            problems.append(tid)

    print()
    if problems:
        print(f"문제 태스크: {problems}")
        return 1
    print("전체 태스크 실제 데이터 로딩 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
