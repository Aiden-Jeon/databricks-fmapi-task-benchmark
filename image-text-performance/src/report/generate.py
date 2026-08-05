"""리포트 생성기 (plan §7). Executive Summary + 정량 + 정성 + 시간/비용.

- run별 markdown 리포트를 reports/<run_id>/report.md로 생성.
- Executive Summary: 하이브리드(D14) — 수치에서 fact sheet를 규칙 추출 → judge가 문단화.
  judge 실패 시 규칙 기반 템플릿 문장으로 fallback(결정론적).
- 시간·비용: SampleResult의 latency + usage를 pricing.yaml로 USD 환산(§10).
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from src.cost.pricing import compute_usd, load_pricing
from src.results import SampleResult

# 갤러리 정답판정에서 HTML 표(TXT-3·IMG-6)를 Cell-F1로 재사용. 모듈 로드 시 1회 import
# (per-sample import 비용 제거). txt_3가 fuzzywuzzy 등에 의존하므로 실패 시 None 폴백.
try:
    from src.tasks.txt_3 import cell_f1_score as _cell_f1, parse_html_table as _parse_html_table
except Exception:  # pragma: no cover - 의존성 없으면 HTML 정답판정만 비활성
    _cell_f1 = None
    _parse_html_table = None

# 태스크별 "대표 메트릭" 우선순위 (그래프·Executive Summary 공용).
# 각 태스크 유형의 핵심 지표가 앞에 오도록: 분류 accuracy/f1, 문서QA anls(DocVQA 공식),
# QA token_f1, 표 cell_f1, 캡션 caption_token_f1, 태그·키워드 micro_f1/f1, 요약 rouge1, judge.
# anls는 TXT-1만 내보내므로 token_f1보다 앞에 두면 TXT-1의 대표값이 공식 메트릭이 된다
# (다른 태스크는 anls 키가 없어 영향 없음).
_METRIC_PRIORITY = [
    "accuracy", "f1", "anls", "token_f1", "cell_f1", "caption_token_f1",
    "micro_f1", "rouge1", "precision", "judge_mean", "judge_score_mean",
]


def generate_report(
    run_dir: Path,
    results: list[SampleResult],
    scores: dict[str, Any],
    models_cfg,
) -> Path:
    """run 디렉터리에 report.md를 생성하고 경로를 반환."""
    reports_dir = Path("reports") / run_dir.name
    reports_dir.mkdir(parents=True, exist_ok=True)

    # model_id → Databricks model name(endpoint) 매핑
    ep = {m.id: m.endpoint for m in models_cfg.models}
    modes = list(models_cfg.reasoning_modes)

    task_labels = _load_task_labels()
    perf = _perf_by_model(results, ep)
    facts = _extract_facts(scores, perf)
    summary = _executive_summary(facts, models_cfg)

    # 비교 그래프 생성 (png) → 리포트에 임베드
    charts = _make_charts(reports_dir, scores, perf, ep)

    md = []
    md.append(f"# 벤치마크 리포트 — {run_dir.name}\n")

    # 평가 대상 모델 — **이 run에서 실제로 실행된 모델만** 싣는다.
    # 예전엔 config의 모든 모델을 나열해, `--models sol`처럼 일부만 돌려도 리포트에는
    # 3개가 다 있는 것처럼 보였다(수치가 없는 모델이 표에 등장 → 실행 범위 오해).
    ran_ids = {e.get("model_id") for e in scores.values() if e.get("model_id")}
    ran_models = [m for m in models_cfg.models if m.id in ran_ids] or list(models_cfg.models)
    md.append("## 평가 대상 모델 (Databricks hosted)\n")
    md.append("| 별칭 | Databricks model name | vision | reasoning 파라미터 | timeout |")
    md.append("|---|---|---|---|---|")
    for m in ran_models:
        rt = m.effective_runtime(models_cfg.runtime)
        # 실제로 보낸 reasoning 파라미터를 그대로 싣는다 — "OFF"라는 서술만으로는 모델별로
        # 무엇을 껐는지(혹은 못 껐는지) 알 수 없다. 빈 dict는 모델 기본값(=ON일 수 있음).
        rp = ", ".join(
            f"`{mode}`: `{m.reasoning_params(mode) or '기본값(모델 정의)'}`" for mode in modes
        )
        md.append(
            f"| {m.id} | `{m.endpoint}` | {'✅' if m.supports('vision') else '❌'} | "
            f"{rp} | {rt.timeout_seconds:g}s |"
        )
    skipped_ids = [m.id for m in models_cfg.models if m.id not in ran_ids]
    md.append(f"\n> Judge: `{models_cfg.judge}`\n")
    if skipped_ids and ran_ids:
        md.append(
            f"> 이 run에서 실행되지 않은 설정상의 모델: {', '.join(f'`{i}`' for i in skipped_ids)} "
            f"(`--models` 필터). 아래 수치는 위 표의 모델만 비교한 결과다.\n"
        )

    # Executive Summary는 평가 대상 모델 바로 뒤 (사용자 요구)
    md.append("## Executive Summary\n")
    md.append(summary + "\n")

    # reasoning 정책 텍스트(맨 뒤 '참고'로 붙임)
    if modes == ["minimal"]:
        reasoning_policy = (
            "**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 "
            "개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 "
            "타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 "
            "judge는 완전 OFF가 불가해 지원 최소값을 사용한다."
        )
    else:
        reasoning_policy = f"측정 reasoning 모드: {modes}"

    if charts:
        md.append("## 비교 그래프\n")
        for title, fname in charts:
            md.append(f"### {title}\n")
            md.append(f"![{title}]({fname})\n")
        # 그래프의 태스크 ID 범례 (ID만으로 안 보이므로 설명 병기)
        if task_labels:
            md.append("**태스크 ID 범례:** " + " · ".join(f"{tid}={d}" for tid, d in sorted(task_labels.items())) + "\n")

    md.append("## 정량 결과 (태스크 × 모델)\n")
    md.append(_quant_table(scores, task_labels) + "\n")
    md.append(_scoring_notes(scores) + "\n")

    sig = _significance_table(scores, task_labels)
    if sig:
        md.append("### 통계 유의성 (judge 점수, Wilcoxon signed-rank)\n")
        md.append(sig + "\n")

    md.append("## 성능: 수행시간·비용 (모델별)\n")
    md.append(_perf_table(perf) + "\n")
    md.append(
        "> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). "
        "정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).\n"
    )

    # 정성 갤러리: 모델 간 결과가 갈린 샘플을 태스크별로 (D1)
    sensitive = _load_sensitive_tasks()
    gallery_slides = _gallery_data(results, task_labels, sensitive)
    _attach_gallery_images(gallery_slides, run_dir)  # 실행 시 저장된 이미지 참조(NSFW 제외)
    if gallery_slides:
        md.append("## 정성 비교: 모델 간 결과가 갈린 샘플\n")
        md.append(
            "각 태스크에서 모델들의 출력·판정이 **가장 크게 갈린 샘플**을 골라 나란히 비교한다. "
            "평균 점수로는 안 보이는 모델별 차이를 드러낸다.\n"
        )
        md.append(_gallery_markdown(gallery_slides, reports_dir) + "\n")

    # 고객 설명용 HTML 프레젠테이션 생성 + 리포트에 링크
    pres_path = None
    try:
        from src.report.presentation import build_presentation

        model_dicts = [
            {"id": m.id, "endpoint": m.endpoint, "capabilities": list(m.capabilities)}
            for m in models_cfg.models
        ]
        # 프레젠테이션 맨 뒤 참고 슬라이드에 정책 전문 사용(markdown ** 강조는 제거)
        pres_reasoning = reasoning_policy.replace("**", "").replace("`", "")
        pres_path = build_presentation(
            reports_dir, run_dir.name, model_dicts, models_cfg.judge, pres_reasoning,
            summary, facts, perf, task_labels, charts, gallery_slides,
        )
    except Exception as e:
        print(f"  [프레젠테이션 생성 스킵] {type(e).__name__}: {e}")

    if pres_path:
        # GitHub는 HTML을 소스로만 보여주므로, htmlpreview.github.io로 감싼 "바로 보기" 링크를
        # 함께 제공(클릭 한 번에 렌더). 로컬/소스 링크도 병기.
        preview = _htmlpreview_url(run_dir.name, pres_path.name)
        banner = f"> 📊 고객 설명용 프레젠테이션: "
        if preview:
            banner += f"**[▶ 브라우저로 바로 보기]({preview})** · [HTML 소스](./{pres_path.name})"
        else:
            banner += f"**[HTML 열기](./{pres_path.name})**"
        banner += " — 이 리포트 결과를 슬라이드로 정리\n"
        md.insert(1, banner)

    # Reasoning 정책은 맨 뒤 '참고'로 (사용자 요구 — 앞은 결과 중심)
    md.append("## 참고: Reasoning 정책\n")
    md.append(reasoning_policy + "\n")

    md.append("## Fact Sheet (Executive Summary 근거 — 감사용)\n")
    md.append("```json\n" + json.dumps(facts, ensure_ascii=False, indent=2, default=str) + "\n```\n")

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    # fact sheet 별도 저장(문장-수치 대응 감사)
    (reports_dir / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report_path


# 차트 X축용 짧은 영문 태스크명(한글 desc는 matplotlib 폰트 깨짐 → ASCII로 자립적 라벨).
# 그래프만 봐도 어떤 실험인지 알 수 있게 ID와 함께 표기(요구사항).
_TASK_SHORT_EN = {
    "IMG-1": "caption", "IMG-2": "tags", "IMG-3": "weapon", "IMG-4": "nsfw",
    "IMG-5": "person", "IMG-6": "table-img", "TXT-1": "doc-qa", "TXT-2": "table-qa",
    "TXT-3": "table-struct", "TXT-4": "ko-qa", "TXT-5": "summary", "TXT-6": "sentiment",
    "TXT-7": "keyphrase", "TXT-8": "toxicity",
}


def _make_charts(reports_dir: Path, scores: dict, perf: dict, ep: dict) -> list[tuple[str, str]]:
    """비교 그래프 png 생성 → [(제목, 파일명)]. matplotlib 없거나 실패 시 빈 리스트.

    X축은 'ID + 영문 항목명'으로 라벨링해 그래프만 봐도 어떤 실험인지 알 수 있게 한다
    (한글 desc는 폰트 깨짐 회피 위해 _TASK_SHORT_EN 매핑 사용).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception:
        return []

    made: list[tuple[str, str]] = []

    # === 그래프 1: 태스크별 모델 대표점수 막대 (그룹) ===
    # scores에서 (task_id, model_id) → 대표 메트릭 추출
    metric_priority = _METRIC_PRIORITY
    task_model_score: dict[str, dict[str, float]] = {}
    for _, e in scores.items():
        m = e.get("metrics", {})
        if not isinstance(m, dict) or "error" in m:
            continue
        val = next((m[k] for k in metric_priority if isinstance(m.get(k), (int, float))), None)
        if val is None:
            continue
        task_model_score.setdefault(e["task_id"], {})[e["model_id"]] = float(val)

    if task_model_score:
        try:
            tasks = sorted(task_model_score)
            model_ids = sorted({mid for mv in task_model_score.values() for mid in mv})
            import numpy as np
            x = np.arange(len(tasks))
            width = 0.8 / max(1, len(model_ids))
            fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.7), 4.5))
            for i, mid in enumerate(model_ids):
                vals = [task_model_score[t].get(mid, 0) for t in tasks]
                ax.bar(x + i * width, vals, width, label=mid)
            ax.set_xticks(x + width * (len(model_ids) - 1) / 2)
            # X축: 'IMG-1\ncaption'처럼 ID+영문명 2줄 → 그래프만 봐도 어떤 태스크인지 명확
            xlabels = [f"{t}\n{_TASK_SHORT_EN.get(t, '')}" for t in tasks]
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("representative score (0-1)")
            ax.set_title("Task performance by model (higher=better)")
            ax.legend(title="model", fontsize=8)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(reports_dir / "chart_scores.png", dpi=110)
            plt.close(fig)
            made.append(("태스크별 모델 성능 비교", "chart_scores.png"))
        except Exception:
            pass

    # === 그래프 2: 모델별 latency(median) vs 비용(USD) 이중축 ===
    if perf:
        try:
            models = sorted(perf)
            lat = [perf[m].get("latency_ms_median") or 0 for m in models]
            # 단가 미등록(None)은 0으로 그린다 — 차트에서 막대가 없는 것이 "저렴"이 아니라
            # "계산 불가"임은 표·요약이 명시한다(차트에 텍스트를 넣을 자리가 없다).
            usd = [perf[m].get("total_usd") or 0 for m in models]
            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            x = range(len(models))
            ax1.bar([i - 0.2 for i in x], lat, 0.4, color="steelblue", label="latency median (ms)")
            ax1.set_ylabel("latency median (ms)", color="steelblue")
            ax1.set_xticks(list(x))
            ax1.set_xticklabels(models)
            ax2 = ax1.twinx()
            ax2.bar([i + 0.2 for i in x], usd, 0.4, color="indianred", label="total USD")
            ax2.set_ylabel("total cost (USD)", color="indianred")
            ax1.set_title("Speed vs Cost by model")
            fig.tight_layout()
            fig.savefig(reports_dir / "chart_perf.png", dpi=110)
            plt.close(fig)
            made.append(("모델별 속도 vs 비용", "chart_perf.png"))
        except Exception:
            pass

    return made


def _perf_by_model(results: list[SampleResult], endpoints: dict[str, str]) -> dict[str, dict]:
    """모델별 latency(median/p95)·토큰·USD 비용 집계.

    **pricing 누락을 $0으로 두지 않는다.** `compute_usd`는 pricing.yaml에 엔드포인트가
    없으면 None을 돌려주는데, 예전엔 `if usd:`로 조용히 건너뛰어 그 모델의 total_usd가
    0이 됐다. 그러면 새로 추가한 모델이 단가를 안 채운 것만으로 **"가장 저렴한 모델"로
    선정**돼 리포트가 정반대 결론을 낸다. 여기서는 누락 호출 수를 세고
    `pricing_missing=True`로 표시해, 비용 비교·cheapest 선정에서 제외되게 한다.
    """
    pricing = load_pricing()
    agg: dict[str, dict] = {}
    for r in results:
        a = agg.setdefault(r.model_id, {
            "latencies": [], "usd": 0.0, "in_tok": 0, "out_tok": 0,
            "n": 0, "errors": 0, "priced": 0, "unpriced": 0,
        })
        a["n"] += 1
        if r.finish_reason == "error":
            a["errors"] += 1
            continue
        a["latencies"].append(r.latency_ms_local)
        ep = endpoints.get(r.model_id, "")
        usd = compute_usd(ep, r.usage or {}, pricing)
        if usd is None:
            a["unpriced"] += 1        # pricing.yaml에 엔드포인트 없음
        else:
            a["usd"] += usd
            a["priced"] += 1
        a["in_tok"] += (r.usage or {}).get("prompt_tokens", 0) or 0
        a["out_tok"] += (r.usage or {}).get("completion_tokens", 0) or 0

    out = {}
    for mid, a in agg.items():
        lat = a["latencies"]
        # 단가를 못 찾은 호출이 하나라도 있으면 비용을 신뢰할 수 없다(부분 합계는 과소계상).
        pricing_missing = a["unpriced"] > 0
        entry = {
            "n_calls": a["n"],
            "errors": a["errors"],
            "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(_p95(lat), 1) if lat else None,
            # 누락이면 None — 0으로 두면 "가장 저렴"으로 오선정된다.
            "total_usd": None if pricing_missing else round(a["usd"], 6),
            "in_tokens": a["in_tok"],
            "out_tokens": a["out_tok"],
        }
        if pricing_missing:
            entry["pricing_missing"] = True
            entry["unpriced_calls"] = a["unpriced"]
            entry["endpoint"] = endpoints.get(mid, "")
        out[mid] = entry
    return out


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def _extract_facts(scores: dict[str, Any], perf: dict[str, dict]) -> dict[str, Any]:
    """수치에서 Executive Summary용 핵심 사실을 규칙 기반으로 추출(D14)."""
    # 태스크별 대표 점수(모델→점수) 뽑기 — 대표 메트릭 우선순위
    metric_priority = _METRIC_PRIORITY
    per_task: dict[str, dict[str, float]] = {}
    excluded: list[str] = []
    for _, entry in scores.items():
        m = entry["metrics"]
        if not isinstance(m, dict) or "error" in m:
            continue
        # 호출이 절반 이상 실패한 셀은 점수가 장애의 산물이라 순위·요약에서 제외한다.
        # (남겨두면 "성능 낮음"으로 집계돼 Executive Summary가 사실과 다른 결론을 낸다 —
        #  실측: IMG-6 opus 19/30 실패 시 sol이 이긴 것처럼 보였으나 재실행하면 opus 승.)
        if entry.get("unreliable"):
            excluded.append(f"{entry['task_id']}/{entry['model_id']}")
            continue
        val = next((m[k] for k in metric_priority if isinstance(m.get(k), (int, float))), None)
        if val is None:
            continue
        key = f"{entry['task_id']}/{entry['reasoning_mode']}"
        per_task.setdefault(key, {})[entry["model_id"]] = round(float(val), 4)

    # 태스크별 1위 모델. **동점은 공동 1위**로 둔다 — 옛 구현의 max(mv, key=mv.get)는
    # 동점일 때 dict 첫 모델(순회 순서상 대개 opus)에만 승리를 줘서 1위 횟수가 부풀었다.
    winners: dict[str, list[str]] = {}
    for tk, mv in per_task.items():
        if not mv:
            continue
        top = max(mv.values())
        winners[tk] = sorted(k for k, v in mv.items() if v == top)

    # 모델별 1위 횟수(공동 1위는 각자 1회). 동점 태스크 수도 함께 남겨 해석을 돕는다.
    win_counts: dict[str, int] = {}
    n_ties = 0
    for ws in winners.values():
        if len(ws) > 1:
            n_ties += 1
        for w in ws:
            win_counts[w] = win_counts.get(w, 0) + 1

    # 비용·속도 최고/최저. **단가 누락 모델은 cheapest 후보에서 제외**한다 —
    # total_usd=None(=pricing.yaml 미등록)을 0으로 취급하면 단가를 안 채운 새 모델이
    # 자동으로 "가장 저렴한 모델"이 돼 리포트가 정반대 결론을 낸다.
    priced = [k for k in perf if isinstance(perf[k].get("total_usd"), (int, float))]
    cheapest = min(priced, key=lambda k: perf[k]["total_usd"]) if priced else None
    unpriced_models = sorted(k for k in perf if perf[k].get("pricing_missing"))
    fastest = min(
        (k for k in perf if perf[k]["latency_ms_median"] is not None),
        key=lambda k: perf[k]["latency_ms_median"],
        default=None,
    )
    return {
        "per_task_scores": per_task,
        "task_winners": winners,          # {task/mode: [공동 1위 모델...]}
        "win_counts": win_counts,         # 공동 1위는 각 모델에 1회씩
        "n_tied_tasks": n_ties,
        "excluded_unreliable": excluded,  # 실패율 과다로 순위에서 뺀 셀
        "cheapest_model": cheapest,
        "fastest_model": fastest,
        "unpriced_models": unpriced_models,   # pricing.yaml 미등록 → 비용 비교 불가
        "perf": perf,
    }


def _executive_summary(facts: dict[str, Any], models_cfg) -> str:
    """하이브리드 요약(D14): judge로 문단화 시도, 실패 시 규칙 기반 fallback."""
    rule_based = _rule_based_summary(facts)
    # judge 문단화 시도 (실패해도 규칙 기반으로 진행)
    try:
        from src.adapters.fmapi import FMAPIClient, build_text_message

        prompt = (
            "다음은 LLM 벤치마크 집계 사실이다. 이 수치에만 근거해, 어느 모델이 무엇을 잘하고/못하는지, "
            "속도·비용 트레이드오프는 어떤지를 3~4문장 한국어 요약으로 써라. 수치를 지어내지 말 것.\n\n"
            + json.dumps(facts, ensure_ascii=False, default=str)[:3000]
        )
        # judge(gemini)는 reasoning을 완전히 못 끄는 모델 → 사고 토큰에 소진돼 요약이 잘리지
        # 않도록 max_tokens를 크게(3000) 잡는다. timeout도 넉넉히.
        with FMAPIClient(profile=models_cfg.profile, timeout_seconds=max(60, models_cfg.runtime.timeout_seconds)) as c:
            resp = c.chat(models_cfg.judge, build_text_message(prompt), max_tokens=3000)
        if resp.text.strip():
            return resp.text.strip() + f"\n\n<sub>규칙 기반 요약(대조용): {rule_based}</sub>"
    except Exception:
        pass
    return rule_based


def _rule_based_summary(facts: dict[str, Any]) -> str:
    """수치에서 직접 조립하는 결정론적 요약(fallback)."""
    wc = facts.get("win_counts", {})
    parts = []
    if wc:
        # 1위 횟수 최댓값이 여럿이면 단독 1위라고 쓰지 않는다(동점을 공동 1위로 세므로 발생 가능).
        best = max(wc.values())
        tops = sorted(k for k, v in wc.items() if v == best)
        counts = ", ".join(f"{k} {v}회" for k, v in sorted(wc.items(), key=lambda x: -x[1]))
        if len(tops) == 1:
            parts.append(f"태스크별 1위 횟수는 {counts}로 **{tops[0]}**가 가장 많다.")
        else:
            parts.append(f"태스크별 1위 횟수는 {counts}로 **{', '.join(tops)}**가 공동 최다다.")
    if facts.get("n_tied_tasks"):
        parts.append(f"이 중 {facts['n_tied_tasks']}개 태스크는 동점이라 공동 1위로 집계했다.")
    if facts.get("excluded_unreliable"):
        # 제외 사실을 요약에 남긴다 — 조용히 빼면 "왜 이 태스크가 없나"를 알 수 없다.
        parts.append(
            f"호출 실패가 과다한 셀 {len(facts['excluded_unreliable'])}개"
            f"({', '.join(facts['excluded_unreliable'])})는 점수가 장애의 산물이라 순위에서 제외했다."
        )
    if facts.get("fastest_model"):
        f = facts["fastest_model"]
        parts.append(f"응답 속도는 **{f}**가 가장 빠르다(median {facts['perf'][f]['latency_ms_median']}ms).")
    if facts.get("cheapest_model"):
        ch = facts["cheapest_model"]
        parts.append(f"비용은 **{ch}**가 가장 낮다(${facts['perf'][ch]['total_usd']}).")
    if facts.get("unpriced_models"):
        # 단가 누락을 요약에 명시한다 — 조용히 빼면 "왜 이 모델이 비용 비교에 없나"를 알 수 없다.
        parts.append(
            f"단, {', '.join(facts['unpriced_models'])}는 `pricing.yaml`에 단가가 없어 "
            f"비용을 계산하지 못했고 비용 비교에서 제외했다."
        )
    return " ".join(parts) if parts else "집계할 결과가 없습니다."


def _significance_table(scores: dict[str, Any], task_labels: dict[str, str]) -> str:
    """모델 쌍별 judge 점수 차이의 유의성(Wilcoxon signed-rank).

    plan §7·README가 "통계 유의성 병기"를 명시하는데 `wilcoxon_test()`가 구현만 되고
    리포트에서 호출되지 않았다(이슈 F). 생성 태스크의 judge 점수는 샘플당 값이 남아
    있어(judge_detail.scores) 짝지어 검정할 수 있다.

    per-sample 점수가 있는 판정만 다룬다 — 정량 메트릭은 셀 단위 평균만 저장돼
    (스트리밍 O(1) 설계) 짝지을 수 없다. 그 사실을 표 아래에 명시한다.
    """
    from src.scoring.stats import wilcoxon_test

    # (task, mode) → {model: [per-sample judge 점수]}
    by_task: dict[tuple[str, str], dict[str, list]] = {}
    for e in scores.values():
        m = e.get("metrics")
        if not isinstance(m, dict) or e.get("unreliable"):
            continue
        arr = (m.get("judge_detail") or {}).get("scores")
        if isinstance(arr, list) and any(x is not None for x in arr):
            by_task.setdefault((e["task_id"], e["reasoning_mode"]), {})[e["model_id"]] = arr

    rows = []
    for (tid, mode), per_model in sorted(by_task.items()):
        models = sorted(per_model)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                # judge 실패(None)가 있는 샘플은 짝을 만들 수 없어 제외한다.
                pairs = [
                    (x, y) for x, y in zip(per_model[a], per_model[b])
                    if x is not None and y is not None
                ]
                if len(pairs) < 2:
                    continue
                res = wilcoxon_test([x for x, _ in pairs], [y for _, y in pairs])
                mean_a = sum(x for x, _ in pairs) / len(pairs)
                mean_b = sum(y for _, y in pairs) / len(pairs)
                pval = res.get("pval")
                if pval is None:
                    verdict = "판정 불가(차이 없음/표본 부족)"
                elif res.get("significant"):
                    winner = a if mean_a > mean_b else b
                    verdict = f"**유의** (p={pval:.4f}) → {winner} 우세"
                else:
                    verdict = f"유의하지 않음 (p={pval:.4f})"
                desc = task_labels.get(tid, "")
                label = f"{tid} · {desc}" if desc else tid
                rows.append(
                    f"| {label} | {a} vs {b} | {mean_a:.2f} vs {mean_b:.2f} | {len(pairs)} | {verdict} |"
                )

    if not rows:
        return ""
    header = [
        "| 태스크 | 모델 쌍 | judge 평균 | n(짝) | 판정 |",
        "|---|---|---|---|---|",
    ]
    footer = (
        "\n> Wilcoxon signed-rank(양측, α=0.05). **judge 점수에만** 적용한다 — 정량 메트릭은 "
        "셀 단위 평균만 저장해(스트리밍 O(1) 설계) 샘플을 짝지을 수 없다. "
        "'유의하지 않음'은 두 모델이 같다는 뜻이 아니라 이 표본에서 차이를 확인할 수 없다는 뜻이다."
    )
    return "\n".join(header + rows) + "\n" + footer


def _scoring_notes(scores: dict[str, Any]) -> str:
    """정량표 아래에 붙는 채점 조건 주석.

    수치를 어떤 조건으로 냈는지(한국어 토큰화 백엔드, 실패 처리 규칙)를 표 바로 옆에
    둔다. 이 정보가 멀리 떨어져 있으면 표만 보고 해석해 오독이 생긴다 — 실측 사고:
    음절 폴백으로 채점된 TXT-4 token_f1을 형태소 기준으로 읽었고, 호출 실패로 0점이
    된 IMG-6를 성능으로 읽었다.
    """
    notes = ["> **채점 조건**"]

    # 한국어 토큰화 백엔드: 형태소(mecab) vs 음절 폴백 — 한국어 점수 해석이 달라진다.
    # 'unknown'은 tokenize(ko) 호출 전에 읽힌 값(그 셀은 한국어 채점을 안 했다는 뜻)이라 제외.
    backends = {
        m["korean_backend"]
        for e in scores.values()
        if isinstance(m := e.get("metrics"), dict)
        and isinstance(m.get("korean_backend"), str)
        and m["korean_backend"] != "unknown"
    }
    if backends:
        b = ", ".join(sorted(backends))
        if backends == {"mecab"}:
            notes.append(f"> - 한국어 토큰화: **형태소(mecab)** — ROUGE·Token-F1이 형태소 기준이다.")
        else:
            notes.append(
                f"> - 한국어 토큰화: **{b}** — `syllable`은 mecab 미설치 시 음절 단위 폴백이라 "
                f"형태소 기준이 아니다(점수가 관대해질 수 있음)."
            )

    # 표본 부족(요청보다 적게 로드된 태스크)
    short = {
        e["task_id"]: (e.get("n"), e.get("n_requested"))
        for e in scores.values()
        if e.get("n_requested") and (e.get("n") or 0) < e["n_requested"]
    }
    if short:
        detail = ", ".join(f"{t} {got}/{want}" for t, (got, want) in sorted(short.items()))
        notes.append(
            f"> - ⚠️ 표본 부족: {detail}. 데이터셋이 요청 수만큼 주지 못한 태스크로, 다른 태스크와 "
            f"같은 신뢰도로 비교하면 안 된다."
        )

    # 실패 처리 규칙
    n_unreliable = sum(1 for e in scores.values() if e.get("unreliable"))
    n_failed_cells = sum(1 for e in scores.values() if e.get("n_call_failed"))
    if n_failed_cells:
        notes.append(
            f"> - 호출 실패: {n_failed_cells}개 셀에 실패가 있다(위 '실패' 열). 실패한 샘플은 "
            f"**채점에서 제외**하므로(0점으로 세지 않음) 그 셀의 점수는 성공한 샘플 기준이다 — "
            f"표의 `n_evaluated`가 요청 샘플 수보다 작은 이유다. 실패는 엔드포인트 문제이지 "
            f"모델 성능이 아니다."
        )
    if n_unreliable:
        notes.append(
            f"> - ⚠️ 실패율 50% 초과 **{n_unreliable}개 셀은 순위·요약에서 제외**했다"
            f"(점수가 장애의 산물). `timeout_seconds` 상향 후 재실행 권장."
        )
    notes.append(
        "> - judge 실패(응답 잘림·형식 이탈)는 해당 샘플을 평균에서 **제외**하고 위 표에 건수를 "
        "표기한다. 중간값으로 메우지 않는다."
    )
    return "\n".join(notes)


def _quant_table(scores: dict[str, Any], task_labels: dict[str, str]) -> str:
    """태스크×모델×모드 점수 표(markdown). 태스크는 'ID · 설명'으로 사람이 구분 가능하게.

    **실패 열을 반드시 함께 보여준다.** 호출이 실패하면 그 샘플은 0점으로 채점되는데
    예외가 아니라서 점수만 보면 "성능이 낮다"로 읽힌다(실측: IMG-6 opus 19/30 타임아웃
    실패 → cell_f1 0.290, 실제는 0.841). judge 실패도 같은 이유로 노출한다.
    """
    rows = [
        "| 태스크 | 모델 | reasoning | 대표 메트릭 | 실패 |",
        "|---|---|---|---|---|",
    ]
    for _, e in sorted(scores.items()):
        m = e["metrics"]
        if isinstance(m, dict) and "error" not in m:
            disp = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            cell = ", ".join(f"{k}={round(v, 3)}" for k, v in list(disp.items())[:6]) or "—"
        else:
            cell = f"오류: {m.get('error', '?')}" if isinstance(m, dict) else "—"
        tid = e["task_id"]
        desc = task_labels.get(tid, "")
        task_col = f"{tid} · {desc}" if desc else tid

        # 실패 요약: 호출 실패(모델 응답 자체 실패) + judge 실패(채점 불가) + 표본 부족을 한 칸에.
        notes = []
        n = e.get("n") or 0
        n_req = e.get("n_requested")
        if n_req and n < n_req:
            # 요청보다 적게 로드된 태스크 — 표본이 작다는 사실이 표에 보여야 한다
            # (실측: TXT-2가 미러 split 문제로 항상 n=10인데 다른 태스크와 같은 무게로 실렸다).
            notes.append(f"표본 {n}/{n_req}")
        n_fail = e.get("n_call_failed") or 0
        if n_fail:
            notes.append(f"호출 {n_fail}/{n}")
        jd = m.get("judge_detail") if isinstance(m, dict) else None
        j_fail = (jd or {}).get("n_failed") or 0
        if j_fail:
            notes.append(f"judge {j_fail}")
        fail_col = ", ".join(notes) if notes else "—"
        if e.get("unreliable"):
            # 절반 이상 실패 → 점수를 성능으로 읽으면 안 된다는 신호를 같은 줄에 박아둔다.
            fail_col = f"⚠️ **{fail_col} — 신뢰불가**"

        rows.append(
            f"| {task_col} | {e['model_id']} | {e['reasoning_mode']} | {cell} | {fail_col} |"
        )
    return "\n".join(rows)


def _load_task_labels(path: str = "config/tasks.yaml") -> dict[str, str]:
    """task_id → desc 매핑 로드 (리포트 표기용). 실패 시 빈 dict."""
    try:
        import yaml

        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        labels = {}
        for t in cfg.get("image_tasks", []) + cfg.get("text_tasks", []):
            labels[t["id"]] = t.get("desc", "")
        return labels
    except Exception:
        return {}


def _perf_table(perf: dict[str, dict]) -> str:
    """모델별 시간·비용 표. 단가 미등록 모델은 비용을 숫자로 쓰지 않고 이유를 적는다."""
    rows = ["| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |",
            "|---|---|---|---|---|---|---|---|"]
    missing: list[str] = []
    for mid, p in sorted(perf.items()):
        if p.get("pricing_missing"):
            # $0으로 두면 "가장 저렴"으로 오독된다 → 원인을 셀에 직접 쓴다.
            usd = "⚠️ 단가 미등록"
            missing.append(f"`{mid}`(endpoint `{p.get('endpoint') or '?'}`)")
        else:
            usd = p["total_usd"]
        rows.append(
            f"| {mid} | {p['n_calls']} | {p['errors']} | {p['latency_ms_median']} | "
            f"{p['latency_ms_p95']} | {p['in_tokens']} | {p['out_tokens']} | {usd} |"
        )
    table = "\n".join(rows)
    if missing:
        table += (
            f"\n\n> ⚠️ **비용 계산 불가**: {', '.join(missing)}가 `config/pricing.yaml`의 "
            f"`models:`에 없습니다. 해당 모델의 DBU 단가(`dbu_in`/`dbu_out`)를 추가하면 비용이 "
            f"계산됩니다. **비용 비교·'가장 저렴한 모델' 선정에서 제외**했습니다"
            f"(0으로 두면 단가 누락이 최저 비용으로 오독됩니다)."
        )
    return table


def _load_sensitive_tasks(path: str = "config/tasks.yaml") -> set[str]:
    """sensitive:true 태스크 id 집합 (IMG-4 NSFW 등 — 입력 숨김)."""
    try:
        import yaml

        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return {
            t["id"]
            for t in cfg.get("image_tasks", []) + cfg.get("text_tasks", [])
            if t.get("sensitive")
        }
    except Exception:
        return set()


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _extract_question(prompt: str, task_id: str) -> str:
    """프롬프트에서 '실제 질문/지시'만 뽑아 갤러리에 별도 표시(사람이 무엇을 물었는지 명확히).

    태스크 프롬프트는 대개 '지시문 … 큰 컨텍스트(표/문서) … 질문'  또는
    '지시문 … 이미지'  구조라, 컨텍스트에 묻혀 질문이 안 보인다(특히 표가 크면 잘림).
    - "Question:"/"질문:" 라벨이 있으면 그 뒤를 우선 사용.
    - 없으면 프롬프트의 첫 지시 문장(명령/의문문)을 사용(이미지 태스크의 "Describe…"/"Extract…" 등).
    실패 시 빈 문자열(호출부에서 생략).
    """
    if not prompt:
        return ""
    # 1) 명시적 질문 라벨(대소문자·한/영). 컨텍스트 뒤에 오는 '마지막' 라벨이 실제 질문
    #    (예: TXT-2는 "answer the question" 지시문 뒤 표, 그 뒤 "Question: …"가 진짜 질문).
    labels = list(re.finditer(r"(?:^|\n)\s*(?:question|질문)\s*[:：]\s*([^\n]+)", prompt, re.IGNORECASE))
    if labels:
        return " ".join(labels[-1].group(1).split())[:300]
    # 2) 라벨이 없으면 첫 비어있지 않은 지시 줄(이미지 태스크의 단일 지시문 등)
    for line in prompt.splitlines():
        s = line.strip()
        if not s:
            continue
        # 표/문서 헤더 라인은 건너뜀
        if s.lower().startswith(("table:", "document:", "context:", "지문:", "문서:", "표:", "|")):
            continue
        return " ".join(s.split())[:300]
    return ""


def _sample_correct(task_id: str, output: str, reference: Any) -> bool | None:
    """갤러리 선택용 per-sample 정답 판정(대략적 — 정확 메트릭이 아니라 케이스 선별용).

    태스크의 정답(reference) 형태로 분기한다:
    - int(이진·다중분류: IMG-3/4/5, TXT-6/8): 출력에서 라벨을 뽑아 일치 여부.
    - list(QA·캡션·태그·키워드: IMG-1/2, TXT-1/2/4/7): 정답 중 하나라도 출력에 포함되면 정답.
    - str HTML(표구조: TXT-3, IMG-6): parse_html_table+cell_f1_score(재사용)로 F1≥0.5면 정답.
    - str 그 외(요약 TXT-5): 명확한 정답 개념이 약함 → None(정답판정 불가, 동점 처리).
    판정 불가/에러 출력은 None. 에러 출력('__ERROR__')은 항상 오답으로 본다.
    """
    if not isinstance(output, str) or output.startswith("__ERROR__"):
        return False if isinstance(output, str) and output.startswith("__ERROR__") else None
    low = output.strip().lower()

    if isinstance(reference, bool):
        reference = int(reference)
    if isinstance(reference, int):
        # 출력에서 이진/분류 라벨 추출. 단어 경계(\b)로 매칭해 'now'가 'no'로,
        # 'cannot'이 오분류되는 것 방지. 부정 우선("no weapon"→0). 긍정 키워드는
        # weapon/threat/nsfw 등 존재 신호(부정과 함께 나오면 부정 우선).
        neg = bool(re.search(r"\b(no|not|none|negative|absent|아니|없)\b", low)) or low in ("no", "0")
        pos_kw = bool(re.search(r"\b(yes|positive|present|긍정|있)\b", low)) or low in ("yes", "1")
        pred = None
        if neg:
            pred = 0                       # 부정 표현 우선(존재 부정)
        elif pos_kw:
            pred = 1
        elif re.search(r"\b1\b", low):
            pred = 1
        elif re.search(r"\b0\b", low):
            pred = 0
        return (pred == int(reference)) if pred is not None else None

    if isinstance(reference, (list, tuple, set)):
        refs = [str(x).strip().lower() for x in reference if str(x).strip()]
        if not refs:
            return None
        # 짧은/숫자 정답은 단어 경계로(‘3’이 ‘30’에 오매칭 방지). 긴 문자열은 부분 포함.
        for r in refs:
            if len(r) <= 4 or r.replace(".", "").replace("-", "").isdigit():
                if re.search(rf"(?<![\w.]){re.escape(r)}(?![\w.])", low):
                    return True
            elif r in low:
                return True
        return False

    if isinstance(reference, str):
        rs = reference.strip()
        if "<t" in rs.lower() and _cell_f1 and _parse_html_table:  # HTML 표(TXT-3, IMG-6)
            try:
                wrap = output if "<table" in low else f"<table>{output}</table>"
                gwrap = rs if "<table" in rs.lower() else f"<table>{rs}</table>"
                return _cell_f1(_parse_html_table(wrap), _parse_html_table(gwrap)) >= 0.5
            except Exception:
                return None
        return None  # 요약 등 자유형 정답 → 판정 불가
    return None


def _gallery_data(
    results: list[SampleResult],
    task_labels: dict[str, str],
    sensitive: set[str],
    per_task: int = 1,
    max_chars: int = 160,
) -> list[dict[str, Any]]:
    """정성 비교 샘플을 태스크별로 선택. **모든 태스크에 최소 1개** 케이스를 보장한다.

    선택 우선순위(요구사항): 모델 간 출력이 갈린 샘플 > 하나라도 오답인 샘플 > 전부 정답인 샘플.
    반환: [{task_id, label, sample_id, reference, sensitive, rows, correctness, tier}]
    - (task, sample_id)로 그룹핑. 에러 행도 '오답'으로 취급해 후보에 남긴다(그래야 IMG-6처럼
      한 모델만 성공한 태스크도 케이스가 생긴다).
    - tier: 'disagree' | 'some_wrong' | 'all_correct' | 'undetermined'(정답 판정 불가 태스크).
    """
    grouped: dict[tuple[str, int], dict[str, tuple[str, Any]]] = {}
    prompts: dict[tuple[str, int], str] = {}
    for r in results:
        # 에러 행도 포함(모델 간 차이·오답 케이스의 근거). prompt는 모델 간 동일.
        grouped.setdefault((r.task_id, r.sample_id), {})[r.model_id] = (r.model_output, r.reference)
        prompts.setdefault((r.task_id, r.sample_id), r.prompt)

    # 태스크별 후보에 tier 점수를 매겨 최선의 per_task개를 고른다.
    by_task: dict[str, list] = {}
    for (task_id, sid), model_out in grouped.items():
        outputs = [o for o, _ in model_out.values()]
        ref = next(iter(model_out.values()))[1]
        uniq = len({_norm(o) for o in outputs})
        # per-sample 정답 판정(모델별). None은 판정 불가.
        corr = {mid: _sample_correct(task_id, o, rf) for mid, (o, rf) in model_out.items()}
        judged = [v for v in corr.values() if v is not None]
        any_wrong = any(v is False for v in judged)
        all_correct = bool(judged) and all(v is True for v in judged)
        disagree = uniq >= 2 and len(model_out) >= 2
        # tier 우선순위 점수(높을수록 우선): 이견 > 오답 > 전부정답 > 판정불가
        if disagree:
            tier, prio = "disagree", 3
        elif any_wrong:
            tier, prio = "some_wrong", 2
        elif all_correct:
            tier, prio = "all_correct", 1
        else:
            tier, prio = "undetermined", 0
        by_task.setdefault(task_id, []).append((prio, uniq, sid, model_out, corr, tier))

    slides: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        # 우선순위(tier) 내림차순, 동순위면 모델 간 고유값 많은 것 우선
        cands = sorted(by_task[task_id], key=lambda x: (-x[0], -x[1]))[:per_task]
        is_image = task_id.startswith("IMG-")
        for prio, uniq, sid, model_out, corr, tier in cands:
            ref = next(iter(model_out.values()))[1]
            rows = [
                (mid, _truncate(str(model_out[mid][0]).replace("\n", " "), max_chars))
                for mid in sorted(model_out)
            ]
            # 모델별 정답 여부(True/False/None) — 표에 ✅/❌로 병기
            correctness = {mid: corr.get(mid) for mid in sorted(model_out)}
            prompt_full = prompts.get((task_id, sid), "")
            slides.append({
                "task_id": task_id,
                "label": task_labels.get(task_id, ""),
                "sample_id": sid,
                "reference": _truncate(str(ref), max_chars),
                "sensitive": task_id in sensitive,
                "is_image": is_image,
                "tier": tier,               # disagree | some_wrong | all_correct | undetermined
                "correctness": correctness,  # {model: True/False/None}
                # 질문/지시: 컨텍스트(표·문서)에 묻히거나 잘려 안 보이는 실제 물음을 별도로 뽑아 항상 표시.
                #            텍스트·이미지 태스크 모두(이미지도 "Describe…"/"Extract…" 지시가 있음).
                "question": _extract_question(prompt_full, task_id),
                # 입력: 텍스트 태스크는 prompt 전문(사람이 직접 판별용, 길이 넉넉히).
                #       이미지 태스크는 질문만 prompt에 있고 이미지는 image_data_uri로 별도 채움.
                "input_text": "" if is_image else _truncate(prompt_full, 1200),
                "image_data_uri": None,   # 이미지 태스크: generate_report에서 재로드해 채움(D3: NSFW 제외)
                "rows": rows,
            })
    return slides


def _attach_gallery_images(slides: list[dict[str, Any]], run_dir: Path) -> None:
    """갤러리 샘플의 이미지를 **실행 시 저장된 파일**(run_dir/images/)에서 읽어 data URI로 채운다.

    runner가 실행 시점에 `images/<task>_s<sid>.jpg`로 저장하므로 sample_id ↔ 이미지가 정확
    (streaming 재로드의 비재현성 버그 방지). sensitive(NSFW)는 저장 안 됨 → 자동 스킵.
    파일이 없으면(과거 run·저장 실패) 조용히 건너뜀.
    """
    import base64

    img_dir = Path(run_dir) / "images"
    if not img_dir.exists():
        return
    for g in slides:
        if not g.get("is_image") or g.get("sensitive"):
            continue
        p = img_dir / f"{g['task_id']}_s{g['sample_id']}.jpg"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            g["image_data_uri"] = f"data:image/jpeg;base64,{b64}"


def _gallery_markdown(slides: list[dict[str, Any]], reports_dir: Path | None = None) -> str:
    """_gallery_data 결과를 markdown으로. 입력(텍스트/이미지)을 함께 표시해 사람이 직접 판별 가능.

    - 텍스트 태스크: 입력 텍스트를 인용 블록으로.
    - 이미지 태스크: image_data_uri를 png로 저장(reports_dir)하고 링크(NSFW는 미표시).
    """
    import base64

    if not slides:
        return ""
    by_task: dict[str, list] = {}
    for g in slides:
        by_task.setdefault(g["task_id"], []).append(g)
    blocks: list[str] = []
    for task_id in sorted(by_task):
        blocks.append(f"### {task_id} · {by_task[task_id][0]['label']}\n")
        for g in by_task[task_id]:
            tag = " (민감 태스크 — 입력 비표시, 판정값만)" if g["sensitive"] else ""
            ref = str(g["reference"])
            # 정답: 여러 줄·HTML(예: TXT-3 <table>)은 인라인 코드가 깨지므로 코드펜스로.
            # 코드펜스 안에서는 raw HTML 태그가 렌더되지 않아 안전(GitHub GFM).
            # tier 라벨: 왜 이 샘플을 골랐는지(요구사항: 이견>오답>전부정답 우선)
            tier_label = {
                "disagree": "모델 간 판정이 갈린 케이스",
                "some_wrong": "일부 모델 오답 케이스",
                "all_correct": "전부 정답 케이스(이견·오답 없음)",
                "undetermined": "예시 케이스(자동 정답판정 불가 태스크)",
            }.get(g.get("tier", ""), "")
            if "\n" in ref or "<" in ref:
                blocks.append(f"**샘플 #{g['sample_id']}**{tag} · _{tier_label}_ · 정답:\n\n{_fence(ref)}\n")
            else:
                blocks.append(f"**샘플 #{g['sample_id']}**{tag} · _{tier_label}_ · 정답: `{ref}`\n")
            # 질문/지시를 별도로 명확히 표시 (컨텍스트에 묻히거나 잘려 안 보이는 문제 해결).
            q = g.get("question")
            if q:
                blocks.append(f"**질문/지시:** {_cell(q)}\n")
            # 입력 표시 (사람이 직접 판별용). 인용문(>)은 raw HTML 태그(<table> 등)를 렌더해
            # 표가 깨지므로, HTML/태그가 있으면 코드펜스로. 없으면 인용문.
            if g.get("input_text"):
                inp = g["input_text"].strip()
                if "<" in inp:
                    blocks.append(f"**입력:**\n\n{_fence(inp)}\n")
                else:
                    blocks.append("**입력:**\n> " + inp.replace("\n", "\n> ") + "\n")
            elif g.get("image_data_uri") and reports_dir is not None:
                fname = f"gallery_{task_id}_s{g['sample_id']}.png"
                try:
                    b64 = g["image_data_uri"].split(",", 1)[1]
                    (reports_dir / fname).write_bytes(base64.b64decode(b64))
                    blocks.append(f"**입력 이미지:**\n\n![{task_id} sample {g['sample_id']}]({fname})\n")
                except Exception:
                    pass
            elif g.get("is_image") and not g.get("sensitive"):
                blocks.append("_(입력 이미지 없음 — 아래 출력만 참고)_\n")
            # 모델 출력: 표 셀 안이라 개행 제거 + | 이스케이프 + 백틱으로 감싸 HTML 태그 렌더 방지.
            # 정답? 열: 자동 판정 결과(✅정답/❌오답/— 판정불가).
            corr = g.get("correctness", {})
            blocks.append("| 모델 | 정답? | 출력/판정 |")
            blocks.append("|---|:--:|---|")
            for mid, out in g["rows"]:
                mark = {True: "✅", False: "❌"}.get(corr.get(mid), "—")
                blocks.append(f"| {mid} | {mark} | `{_cell(out)}` |")
            blocks.append("")
    return "\n".join(blocks)


def _cell(text: str) -> str:
    """markdown 표 셀 안전화: 개행→공백, 백틱 제거, | 이스케이프."""
    s = str(text).replace("\n", " ").replace("\r", " ").replace("`", "'")
    return s.replace("|", "\\|")


def _fence(text: str) -> str:
    """텍스트를 코드펜스로 감싼다. 내부에 백틱 런이 있으면 더 긴 fence를 써서 안전하게.

    코드펜스 안에서는 raw HTML(<table> 등)이 렌더되지 않아 markdown 붕괴를 막는다.
    """
    import re

    s = str(text)
    longest = max((len(m) for m in re.findall(r"`+", s)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{s}\n{fence}"


def _htmlpreview_url(run_id: str, filename: str) -> str | None:
    """htmlpreview.github.io로 감싼 프레젠테이션 렌더 URL. git remote에서 owner/repo·branch 추출.

    GitHub는 repo의 HTML을 소스로만 보여주므로 이 프록시로 클릭 한 번에 렌더한다.
    비-git·remote 없음·파싱 실패 시 None(로컬 링크로 폴백).
    """
    import re
    import subprocess

    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "main"
        # git@github.com:owner/repo.git  또는  https://github.com/owner/repo.git
        m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", remote)
        if not m:
            return None
        owner, repo = m.group(1), m.group(2)
        # 이 프로젝트는 repo 안의 image-text-performance/ 서브디렉터리에 있음
        path = f"image-text-performance/reports/{run_id}/{filename}"
        blob = f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
        return f"https://htmlpreview.github.io/?{blob}"
    except Exception:
        return None
