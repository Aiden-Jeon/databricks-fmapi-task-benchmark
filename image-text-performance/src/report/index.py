"""reports/index.md 생성 — 전체 run을 시간순으로 인덱싱 (plan §12).

모델을 추가하거나 재실행할 때마다 새 run-id 리포트가 쌓이므로, 전체를 한눈에
볼 인덱스가 필요하다. reports/ 디렉터리를 스캔해 매번 재생성한다(append 아님 →
중복·순서 문제 없음). 각 run의 report 옆 manifest(있으면)에서 요약을 읽는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def rebuild_index(reports_root: str | Path = "reports", results_root: str | Path = "results") -> Path:
    """reports/ 하위 run들을 스캔해 index.md를 재생성. 경로 반환.

    최신 run으로 README의 '최신 리포트' 링크도 자동 갱신(하드코딩 run-id가 삭제되면
    링크가 깨지던 문제 방지 — 옛 run 삭제·새 run 추가와 무관하게 항상 실재하는 최신을 가리킴).
    """
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(reports_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        report_md = d / "report.md"
        if not report_md.exists():
            continue
        # manifest는 results/<run-id>/에 있음 (있으면 요약 사용)
        manifest = _load_manifest(Path(results_root) / d.name)
        runs.append((d.name, manifest))

    lines = ["# 벤치마크 리포트 인덱스\n", f"총 {len(runs)}개 run (최신순).\n"]
    lines.append("| run-id | 일시 | 모델 | reasoning | 태스크수 | 샘플/태스크 | git |")
    lines.append("|---|---|---|---|---|---|---|")
    for run_id, m in runs:
        models = ",".join(x.get("id", "?") for x in m.get("models", [])) or "?"
        modes = ",".join(m.get("reasoning_modes", [])) or "?"
        ntask = len(m.get("task_ids", [])) or "?"
        spt = m.get("samples_per_task", "?")
        git = m.get("git_commit", "?") or "?"
        created = m.get("created_at", "")[:19]
        lines.append(f"| [{run_id}]({run_id}/report.md) | {created} | {models} | {modes} | {ntask} | {spt} | {git} |")

    index_path = reports_root / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # README의 '최신 리포트' 링크를 최신 run으로 갱신(있으면). 스캔 결과 최상단이 최신.
    if runs:
        latest_id, latest_m = runs[0]
        models = ",".join(x.get("id", "?") for x in latest_m.get("models", [])) or "opus·sol·glm"
        spt = latest_m.get("samples_per_task", "?")
        _update_readme_latest(reports_root.parent / "README.md", latest_id, models, spt)

    return index_path


def _update_readme_latest(readme_path: Path, run_id: str, models: str, spt: object) -> None:
    """README의 '최신 리포트' 불릿 한 줄을 최신 run 링크로 치환.

    치환 대상은 `- **[최신 리포트 (…)](./reports/<id>/report.md)**` 패턴 한 줄.
    패턴이 없으면(README 구조 변경 등) 조용히 skip — 인덱스 생성 자체는 실패시키지 않는다.
    """
    if not readme_path.exists():
        return
    try:
        text = readme_path.read_text(encoding="utf-8")
        new_line = (
            f"- **[최신 리포트 ({run_id})](./reports/{run_id}/report.md)** — "
            f"{spt}샘플, {models}. 리포트 내 \"고객 설명용 프레젠테이션\" 배너로 슬라이드(HTML)도 바로 볼 수 있다. "
            f"*(이 줄은 새 run 때 runner가 자동 갱신)*"
        )
        # '최신 리포트' 불릿 한 줄만 교체(줄 단위, 앞의 '- **[최신 리포트'로 시작).
        pattern = re.compile(r"^- \*\*\[최신 리포트 .*$", re.MULTILINE)
        if pattern.search(text):
            new_text = pattern.sub(new_line, text, count=1)
            if new_text != text:
                readme_path.write_text(new_text, encoding="utf-8")
    except Exception:
        # README 갱신 실패가 인덱스 생성을 막지 않도록 조용히 무시.
        pass


def _load_manifest(results_dir: Path) -> dict:
    p = results_dir / "manifest.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}
