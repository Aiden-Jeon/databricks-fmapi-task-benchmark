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

    # 검사 대상은 config에서 유도한다(예전엔 13개를 하드코딩해 IMG-6 추가 시 검사에서 빠졌다).
    order = list(cfgs)
    want = 2
    problems = []
    shortfalls: list[str] = []
    for tid in order:
        cls = registered.get(tid)
        if cls is None:
            print(f"  ❌ {tid}: 미등록(플러그인 없음)")
            problems.append(tid)
            continue
        inst = cls(cfgs.get(tid, {}), registry)
        try:
            samples = inst.load_samples(want, 42)
            n = len(samples)
            # 합성 흔적 탐지: meta에 synthetic/mock/test 표시
            synth = any(
                any(k in str(s.meta).lower() for k in ("synthetic", "mock", "dummy", "합성"))
                for s in samples
            )
            ref_preview = str(samples[0].reference)[:50] if samples else "—"
            # 요청 수보다 적게 로드되면 경고. 리포트의 n이 조용히 작아지는 원인이라
            # (예: 30 요청에 10샘플만) 눈에 보이게 남긴다.
            short = 0 < n < want
            if synth:
                flag = "🔴 합성흔적"
            elif n == 0:
                flag = "⚠️ 0샘플"
            elif short:
                flag = f"⚠️ 부족({n}/{want})"
            else:
                flag = "✅"
            print(f"  {flag} {tid}: {n}샘플 | ref={ref_preview}")
            if synth or n == 0:
                problems.append(tid)
            elif short:
                shortfalls.append(f"{tid}({n}/{want})")
        except Exception as e:
            print(f"  ❌ {tid}: 로드실패 {type(e).__name__}: {str(e)[:80]}")
            problems.append(tid)

    print()
    if shortfalls:
        # 실패는 아니지만(작은 데이터셋일 수 있음) 리포트 n 불일치의 단서 → 반드시 노출.
        print(f"요청보다 적게 로드된 태스크: {', '.join(shortfalls)}")
    if problems:
        print(f"문제 태스크: {problems}")
        return 1
    print(f"전체 {len(order)}개 태스크 실제 데이터 로딩 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
