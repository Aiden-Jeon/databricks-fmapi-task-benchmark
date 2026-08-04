"""고객 설명용 HTML 프레젠테이션 생성.

리포트(report.md)와 같은 데이터로 슬라이드형 단일 HTML을 만든다. 그래프 png는
base64 data URI로 인라인해 **파일 하나만 공유하면 되는** 자체 완결 형태.
generate_report()가 리포트 생성 직후 호출한다.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any


def _img_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _esc(s: Any) -> str:
    return html.escape(str(s))


def build_presentation(
    reports_dir: Path,
    run_id: str,
    models: list[dict[str, str]],
    judge: str,
    reasoning_note: str,
    summary: str,
    facts: dict[str, Any],
    perf: dict[str, dict],
    task_labels: dict[str, str],
    chart_files: list[tuple[str, str]],
    gallery_slides: list[dict[str, Any]],
) -> Path:
    """슬라이드형 HTML 생성 → 경로 반환.

    gallery_slides: [{task_id, label, sample_id, reference, sensitive, rows:[(model, output)]}]
    """
    slides: list[str] = []

    # 슬라이드 1: 타이틀
    slides.append(f"""
    <section class="slide title">
      <h1>이미지·텍스트 LLM 성능 벤치마크</h1>
      <p class="sub">Databricks Foundation Model API · run <code>{_esc(run_id)}</code></p>
      <p class="models">{" · ".join(f'<code>{_esc(m["endpoint"])}</code>' for m in models)}</p>
    </section>""")

    # 슬라이드 2: 평가 개요 (모델·judge·reasoning)
    model_rows = "".join(
        f"<tr><td>{_esc(m['id'])}</td><td><code>{_esc(m['endpoint'])}</code></td>"
        f"<td>{'✅' if 'vision' in m.get('capabilities', []) else '❌'}</td></tr>"
        for m in models
    )
    slides.append(f"""
    <section class="slide">
      <h2>평가 개요</h2>
      <table><thead><tr><th>별칭</th><th>Databricks model name</th><th>vision</th></tr></thead>
      <tbody>{model_rows}</tbody></table>
      <p><b>Judge</b>: <code>{_esc(judge)}</code></p>
    </section>""")

    # 슬라이드 3: Executive Summary (규칙기반 대조용 <sub>...</sub> 꼬리는 고객용에선 제거)
    clean_summary = summary.split("<sub>")[0].strip()
    slides.append(f"""
    <section class="slide">
      <h2>핵심 요약 (Executive Summary)</h2>
      <p class="summary">{_esc(clean_summary)}</p>
    </section>""")

    # 슬라이드 4+: 그래프
    for title, fname in chart_files:
        uri = _img_data_uri(reports_dir / fname)
        if uri:
            slides.append(f"""
    <section class="slide">
      <h2>{_esc(title)}</h2>
      <img class="chart" src="{uri}" alt="{_esc(title)}"/>
    </section>""")

    # 태스크 ID 범례 (그래프 이해용)
    if task_labels:
        legend = "".join(f"<li><code>{_esc(t)}</code> — {_esc(d)}</li>" for t, d in sorted(task_labels.items()))
        slides.append(f"""
    <section class="slide">
      <h2>태스크 안내</h2>
      <ul class="legend">{legend}</ul>
    </section>""")

    # 성능(시간·비용) 슬라이드
    perf_rows = "".join(
        f"<tr><td>{_esc(m)}</td><td>{p.get('latency_ms_median')}</td>"
        f"<td>{p.get('errors')}</td><td>${p.get('total_usd')}</td></tr>"
        for m, p in sorted(perf.items())
    )
    slides.append(f"""
    <section class="slide">
      <h2>속도 · 비용 · 안정성</h2>
      <table><thead><tr><th>모델</th><th>지연 median(ms)</th><th>오류</th><th>총 비용(USD)</th></tr></thead>
      <tbody>{perf_rows}</tbody></table>
      <div class="note">비용은 pricing.yaml DBU 단가 기반 추정.</div>
    </section>""")

    # 정성 비교 슬라이드 (모델 간 갈린 샘플) — 입력(텍스트/이미지)을 함께 보여 사람이 직접 판별
    for g in gallery_slides:
        rows = "".join(
            f"<tr><td class='m'>{_esc(m)}</td><td>{_esc(o)}</td></tr>" for m, o in g["rows"]
        )
        inp = "입력 비표시 (민감 태스크)" if g.get("sensitive") else f"샘플 #{_esc(g['sample_id'])}"
        # 질문/지시를 별도 강조(컨텍스트에 묻히는 문제 해결). 없으면 생략.
        q_html = f'<p class="qask"><strong>질문/지시:</strong> {_esc(g["question"])}</p>' if g.get("question") else ""
        # 입력 렌더: 이미지면 인라인, 텍스트면 인용 박스
        input_html = ""
        if g.get("image_data_uri"):
            input_html = f'<div class="qinput"><img class="qimg" src="{g["image_data_uri"]}" alt="input image"/></div>'
        elif g.get("input_text"):
            input_html = f'<div class="qinput qtext">{_esc(g["input_text"])}</div>'
        slides.append(f"""
    <section class="slide">
      <h2>정성 비교 · {_esc(g['task_id'])} {_esc(g['label'])}</h2>
      <p class="qmeta">{inp} · 정답: <code>{_esc(g['reference'])}</code></p>
      {q_html}
      {input_html}
      <table class="qtable"><thead><tr><th>모델</th><th>출력/판정</th></tr></thead>
      <tbody>{rows}</tbody></table>
    </section>""")

    # 맨 뒤 참고 슬라이드: Reasoning 정책 (사용자 요구 — 앞은 결과 중심, 방법론은 참고로)
    slides.append(f"""
    <section class="slide">
      <h2>참고 · Reasoning 정책</h2>
      <p class="summary">{_esc(reasoning_note)}</p>
    </section>""")

    n = len(slides)
    body = "\n".join(slides)
    doc = _HTML_TEMPLATE.replace("{{SLIDES}}", body).replace("{{N}}", str(n)).replace("{{RUNID}}", _esc(run_id))

    out = reports_dir / "presentation.html"
    out.write_text(doc, encoding="utf-8")
    return out


_HTML_TEMPLATE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LLM 벤치마크 — {{RUNID}}</title>
<style>
  :root { --fg:#1a1a2e; --accent:#e8500e; --bg:#fdfcfb; --muted:#6b7280; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif;
         color:var(--fg); background:#e9e7e4; }
  .slide { min-height:100vh; padding:6vh 8vw; background:var(--bg); display:flex; flex-direction:column;
           justify-content:center; border-bottom:1px solid #ddd; scroll-snap-align:start; }
  html { scroll-snap-type:y proximity; }
  .title { background:linear-gradient(135deg,#1a1a2e,#3a1c1c); color:#fff; }
  .title h1 { font-size:3rem; margin:0 0 1rem; }
  .title .sub { font-size:1.3rem; color:#f0ede8; }
  .title .models { margin-top:2rem; font-size:1.05rem; color:#e8e6e2; }
  /* 다크 타이틀 슬라이드에서는 code 배경을 투명하게(밝은 베이지 배경이 튀지 않도록) */
  .title code { background:rgba(255,255,255,.14); color:#fff; border:1px solid rgba(255,255,255,.25); }
  h2 { font-size:2rem; color:var(--accent); border-bottom:3px solid var(--accent); padding-bottom:.3rem; }
  table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:1rem; background:#fff; }
  th,td { border:1px solid #ddd; padding:.55rem .8rem; text-align:left; vertical-align:top; }
  th { background:#f3f1ee; }
  td.m, td.m { font-weight:600; white-space:nowrap; }
  code { background:#f0eee9; padding:.1rem .4rem; border-radius:4px; font-size:.9em; }
  .chart { max-width:100%; max-height:70vh; margin:1rem auto; display:block; background:#fff; padding:1rem; border-radius:8px; }
  .summary { font-size:1.3rem; line-height:1.8; max-width:60ch; }
  .legend { columns:2; font-size:1rem; line-height:1.9; }
  .note { color:var(--muted); font-size:.9rem; margin-top:1rem; }
  .qmeta { color:var(--muted); }
  .qinput { margin:1rem 0; }
  .qimg { max-height:38vh; max-width:100%; border:1px solid #ccc; border-radius:8px; background:#fff; }
  .qtext { background:#f3f1ee; border-left:4px solid var(--accent); padding:.8rem 1rem;
           font-size:.95rem; line-height:1.6; max-height:32vh; overflow:auto; white-space:pre-wrap; }
  .qtable td:last-child { font-family:ui-monospace,monospace; font-size:.9rem; }
  .nav { position:fixed; bottom:1rem; right:1rem; background:#1a1a2e; color:#fff; padding:.5rem 1rem;
         border-radius:20px; font-size:.85rem; opacity:.8; }
</style></head>
<body>
{{SLIDES}}
<div class="nav">{{N}} slides · 스크롤로 이동</div>
</body></html>"""
