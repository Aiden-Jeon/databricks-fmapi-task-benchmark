"""리포트 무결성 검증 (커밋 전 필수 게이트).

리포트는 사람이 수치를 읽고 판단하는 산출물이라, 렌더가 깨지거나 실패가 숨으면
잘못된 결론으로 이어진다. 이 프로젝트에서 실제로 났던 사고를 항목으로 고정한다:

1. **GFM 렌더 무결성** — 코드펜스 밖 raw `<table>`이 GitHub에서 렌더되면 표가 깨지고
   다음 섹션을 빨아들인다(TXT-3 옛 버그).
2. **실패 노출** — 호출·judge 실패가 정량표에 보이는지. 안 보이면 장애가 성능으로 읽힌다
   (IMG-6 19/30 실패가 cell_f1 0.290으로 실렸던 사고).
3. **judge_mean=0.0 금지** — 파싱 전멸을 "judge가 최악 평가"로 오독하게 만든다.
4. **셀 수·샘플 수 정합** — scores.json과 samples.jsonl이 맞는지.
5. **backend 표기** — 한국어 점수가 형태소/음절 어느 기준인지 드러나는지.

사용: python3 tests/check_report_integrity.py [run-id]
      (run-id 생략 시 reports/ 아래 최신)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _latest_run() -> str:
    dirs = sorted(d.name for d in Path("reports").glob("*") if d.is_dir())
    if not dirs:
        print("reports/ 아래 run이 없습니다.")
        sys.exit(1)
    return dirs[-1]


def check(run_id: str) -> int:
    report = Path("reports") / run_id / "report.md"
    scores_path = Path("results") / run_id / "scores.json"
    samples_path = Path("results") / run_id / "samples.jsonl"

    problems: list[str] = []
    warnings: list[str] = []

    if not report.exists():
        print(f"❌ 리포트 없음: {report}")
        return 1
    md = report.read_text(encoding="utf-8")
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}

    print(f"검증 대상: {run_id}")
    print("=" * 70)

    # ── 1. 셀·샘플 정합 ────────────────────────────────────────────────
    n_cells = len(scores)
    n_rows = sum(1 for _ in open(samples_path, encoding="utf-8")) if samples_path.exists() else 0
    ns = sorted({v.get("n") for v in scores.values()})
    expected = sum(v.get("n", 0) for v in scores.values())
    print(f"  셀 {n_cells}개 | 샘플 행 {n_rows} (기대 {expected}) | n 분포 {ns}")
    if n_rows != expected:
        problems.append(f"샘플 행 불일치: {n_rows} vs 기대 {expected}")
    err_cells = [k for k, v in scores.items() if "error" in (v.get("metrics") or {})]
    if err_cells:
        problems.append(f"채점 오류 셀: {err_cells}")

    # ── 2. 실패 집계·노출 ──────────────────────────────────────────────
    call_failed = {k: v["n_call_failed"] for k, v in scores.items() if v.get("n_call_failed")}
    unreliable = [k for k, v in scores.items() if v.get("unreliable")]
    judge_failed = {
        k: (v["metrics"].get("judge_detail") or {}).get("n_failed")
        for k, v in scores.items()
        if isinstance(v.get("metrics"), dict)
        and (v["metrics"].get("judge_detail") or {}).get("n_failed")
    }
    print(f"  호출 실패 셀 {len(call_failed)}개 | 신뢰불가 {len(unreliable)}개 | judge 실패 셀 {len(judge_failed)}개")
    if call_failed and "실패" not in md:
        problems.append("호출 실패가 있는데 리포트에 '실패' 표기가 없다")
    for k in unreliable:
        if "신뢰불가" not in md:
            problems.append("unreliable 셀이 있는데 리포트에 '신뢰불가' 경고가 없다")
            break
    if call_failed:
        warnings.append(f"호출 실패 잔존: {call_failed} → 재실행 검토")

    # ── 3. judge_mean=0.0 오독 방지 ────────────────────────────────────
    if "judge_mean=0.0" in md:
        problems.append("judge_mean=0.0이 리포트에 있다(파싱 전멸을 성능으로 오독)")

    # ── 4. 한국어 backend 표기 ─────────────────────────────────────────
    backends = {
        v["metrics"].get("korean_backend")
        for v in scores.values()
        if isinstance(v.get("metrics"), dict) and v["metrics"].get("korean_backend")
    }
    print(f"  한국어 토큰화 backend: {backends or '없음'}")
    if "unknown" in backends:
        problems.append("korean_backend=unknown 셀이 있다(채점 기준 불명 — 재실행 필요)")

    # ── 5. GitHub GFM 렌더 (gh CLI 필요) ───────────────────────────────
    try:
        out = subprocess.run(
            ["gh", "api", "/markdown", "-f", "mode=gfm", "-f", f"text={md}"],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode == 0:
            html = out.stdout
            op, cl = len(re.findall(r"<table", html)), len(re.findall(r"</table>", html))
            h2, h3 = len(re.findall(r"<h2", html)), len(re.findall(r"<h3", html))
            print(f"  GFM 렌더: <table> {op}/{cl} | h2 {h2} h3 {h3}")
            if op != cl:
                problems.append(f"<table> 열림/닫힘 불균형: {op} vs {cl} (표 깨짐)")
            if h2 == 0:
                problems.append("h2 섹션이 없다(리포트 구조 손상)")
        else:
            warnings.append("gh api /markdown 실패 — 렌더 검증 건너뜀")
    except Exception as e:
        warnings.append(f"렌더 검증 건너뜀({type(e).__name__})")

    # ── 6. 필수 산출물 ─────────────────────────────────────────────────
    rd = Path("reports") / run_id
    for f in ("facts.json", "presentation.html"):
        if not (rd / f).exists():
            problems.append(f"산출물 없음: {f}")
    charts = list(rd.glob("chart_*.png"))
    galleries = list(rd.glob("gallery_*.png"))
    print(f"  산출물: chart {len(charts)}개, gallery {len(galleries)}개")
    if not charts:
        problems.append("차트 png가 없다")

    print("=" * 70)
    for w in warnings:
        print(f"⚠️  {w}")
    if problems:
        print(f"\n❌ 문제 {len(problems)}건:")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("\n✅ 리포트 무결성 OK")
    return 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else _latest_run()))
