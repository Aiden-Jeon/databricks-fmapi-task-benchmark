"""리포트 생성기 (plan §7). Executive Summary + 정량 + 정성 + 시간/비용.

- run별 markdown 리포트를 reports/<run_id>/report.md로 생성.
- Executive Summary: 하이브리드(D14) — 수치에서 fact sheet를 규칙 추출 → judge가 문단화.
  judge 실패 시 규칙 기반 템플릿 문장으로 fallback(결정론적).
- 시간·비용: SampleResult의 latency + usage를 pricing.yaml로 USD 환산(§10).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from src.cost.pricing import compute_usd, load_pricing
from src.results import SampleResult

# 태스크별 "대표 메트릭" 우선순위 (그래프·Executive Summary 공용).
# 각 태스크 유형의 핵심 지표가 앞에 오도록: 분류 accuracy/f1, QA token_f1, 표 cell_f1,
# 캡션 caption_token_f1, 태그·키워드 micro_f1/f1, 요약 rouge1, judge.
_METRIC_PRIORITY = [
    "accuracy", "f1", "token_f1", "cell_f1", "caption_token_f1",
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

    # 평가 대상 모델 (Databricks model name 그대로 표기)
    md.append("## 평가 대상 모델 (Databricks hosted)\n")
    md.append("| 별칭 | Databricks model name | vision |")
    md.append("|---|---|---|")
    for m in models_cfg.models:
        md.append(f"| {m.id} | `{m.endpoint}` | {'✅' if m.supports('vision') else '❌'} |")
    md.append(f"\n> Judge: `{models_cfg.judge}`\n")

    # reasoning 정책 (OFF 고정 — 이유 명시)
    md.append("## Reasoning 정책\n")
    if modes == ["minimal"]:
        md.append(
            "**Reasoning OFF(minimal) 단일 모드로 고정.** reasoning의 효과가 특정 태스크 성능 "
            "개선에만 한정되는 반면 실험 시간을 크게 늘리기 때문(특히 GLM은 full reasoning 시 "
            "타임아웃 빈발). 각 모델의 최소 reasoning으로 측정하며, `databricks-claude-opus-5`와 "
            "judge는 완전 OFF가 불가해 지원 최소값을 사용한다.\n"
        )
    else:
        md.append(f"측정 reasoning 모드: {modes}\n")

    md.append("## Executive Summary\n")
    md.append(summary + "\n")

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

    md.append("## 성능: 수행시간·비용 (모델별)\n")
    md.append(_perf_table(perf) + "\n")
    md.append(
        "> 비용은 `config/pricing.yaml` DBU 단가 기반 추정(usd_per_dbu 가정값). "
        "정밀 비용은 `system.ai_gateway.usage` 조인 필요(§10).\n"
    )

    # 정성 갤러리: 모델 간 결과가 갈린 샘플을 태스크별로 (D1)
    sensitive = _load_sensitive_tasks()
    gallery_slides = _gallery_data(results, task_labels, sensitive)
    _attach_gallery_images(gallery_slides)  # 이미지 태스크 샘플 이미지 재로드(NSFW 제외)
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
        reasoning_note = (
            "OFF(minimal) 고정 — reasoning 효과가 특정 태스크에 한정되고 실험 시간을 크게 늘려서."
            if modes == ["minimal"] else f"측정 모드 {modes}"
        )
        pres_path = build_presentation(
            reports_dir, run_dir.name, model_dicts, models_cfg.judge, reasoning_note,
            summary, facts, perf, task_labels, charts, gallery_slides,
        )
    except Exception as e:
        print(f"  [프레젠테이션 생성 스킵] {type(e).__name__}: {e}")

    if pres_path:
        # 링크는 리포트 상단(모델표 다음)이 아니라, 문서 맨 위에 배너로 넣는 게 눈에 띔 →
        # md[1]에 삽입(제목 다음).
        md.insert(1, f"> 📊 **[고객 설명용 프레젠테이션 (HTML)](./{pres_path.name})** — 이 리포트 결과를 슬라이드로 정리\n")

    md.append("## Fact Sheet (Executive Summary 근거 — 감사용)\n")
    md.append("```json\n" + json.dumps(facts, ensure_ascii=False, indent=2, default=str) + "\n```\n")

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    # fact sheet 별도 저장(문장-수치 대응 감사)
    (reports_dir / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report_path


def _make_charts(reports_dir: Path, scores: dict, perf: dict, ep: dict) -> list[tuple[str, str]]:
    """비교 그래프 png 생성 → [(제목, 파일명)]. matplotlib 없거나 실패 시 빈 리스트.

    라벨은 ASCII(태스크 ID·model_id)로 유지해 한글 폰트 깨짐 회피.
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
            ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("representative score (0-1)")
            ax.set_title("Task performance by model")
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
    """모델별 latency(median/p95)·토큰·USD 비용 집계."""
    pricing = load_pricing()
    agg: dict[str, dict] = {}
    for r in results:
        a = agg.setdefault(r.model_id, {"latencies": [], "usd": 0.0, "in_tok": 0, "out_tok": 0, "n": 0, "errors": 0})
        a["n"] += 1
        if r.finish_reason == "error":
            a["errors"] += 1
            continue
        a["latencies"].append(r.latency_ms_local)
        ep = endpoints.get(r.model_id, "")
        usd = compute_usd(ep, r.usage or {}, pricing)
        if usd:
            a["usd"] += usd
        a["in_tok"] += (r.usage or {}).get("prompt_tokens", 0) or 0
        a["out_tok"] += (r.usage or {}).get("completion_tokens", 0) or 0

    out = {}
    for mid, a in agg.items():
        lat = a["latencies"]
        out[mid] = {
            "n_calls": a["n"],
            "errors": a["errors"],
            "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(_p95(lat), 1) if lat else None,
            "total_usd": round(a["usd"], 6),
            "in_tokens": a["in_tok"],
            "out_tokens": a["out_tok"],
        }
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
    for _, entry in scores.items():
        m = entry["metrics"]
        if not isinstance(m, dict) or "error" in m:
            continue
        val = next((m[k] for k in metric_priority if isinstance(m.get(k), (int, float))), None)
        if val is None:
            continue
        key = f"{entry['task_id']}/{entry['reasoning_mode']}"
        per_task.setdefault(key, {})[entry["model_id"]] = round(float(val), 4)

    # 태스크별 1위 모델
    winners = {}
    for tk, mv in per_task.items():
        if mv:
            winners[tk] = max(mv, key=mv.get)

    # 모델별 1위 횟수
    win_counts: dict[str, int] = {}
    for w in winners.values():
        win_counts[w] = win_counts.get(w, 0) + 1

    # 비용·속도 최고/최저
    cheapest = min(perf, key=lambda k: perf[k]["total_usd"]) if perf else None
    fastest = min(
        (k for k in perf if perf[k]["latency_ms_median"] is not None),
        key=lambda k: perf[k]["latency_ms_median"],
        default=None,
    )
    return {
        "per_task_scores": per_task,
        "task_winners": winners,
        "win_counts": win_counts,
        "cheapest_model": cheapest,
        "fastest_model": fastest,
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
        top = max(wc, key=wc.get)
        parts.append(f"태스크별 1위 횟수는 {', '.join(f'{k} {v}회' for k, v in sorted(wc.items(), key=lambda x: -x[1]))}로 **{top}**가 가장 많다.")
    if facts.get("fastest_model"):
        f = facts["fastest_model"]
        parts.append(f"응답 속도는 **{f}**가 가장 빠르다(median {facts['perf'][f]['latency_ms_median']}ms).")
    if facts.get("cheapest_model"):
        ch = facts["cheapest_model"]
        parts.append(f"비용은 **{ch}**가 가장 낮다(${facts['perf'][ch]['total_usd']}).")
    return " ".join(parts) if parts else "집계할 결과가 없습니다."


def _quant_table(scores: dict[str, Any], task_labels: dict[str, str]) -> str:
    """태스크×모델×모드 점수 표(markdown). 태스크는 'ID · 설명'으로 사람이 구분 가능하게."""
    rows = ["| 태스크 | 모델 | reasoning | 대표 메트릭 |", "|---|---|---|---|"]
    for _, e in sorted(scores.items()):
        m = e["metrics"]
        if isinstance(m, dict) and "error" not in m:
            disp = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            cell = ", ".join(f"{k}={round(v, 3)}" for k, v in list(disp.items())[:4]) or "—"
        else:
            cell = f"오류: {m.get('error', '?')}" if isinstance(m, dict) else "—"
        tid = e["task_id"]
        desc = task_labels.get(tid, "")
        task_col = f"{tid} · {desc}" if desc else tid
        rows.append(f"| {task_col} | {e['model_id']} | {e['reasoning_mode']} | {cell} |")
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
    rows = ["| 모델 | 호출 | 오류 | latency median(ms) | p95(ms) | 입력토큰 | 출력토큰 | 비용(USD) |",
            "|---|---|---|---|---|---|---|---|"]
    for mid, p in sorted(perf.items()):
        rows.append(
            f"| {mid} | {p['n_calls']} | {p['errors']} | {p['latency_ms_median']} | "
            f"{p['latency_ms_p95']} | {p['in_tokens']} | {p['out_tokens']} | {p['total_usd']} |"
        )
    return "\n".join(rows)


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


def _gallery_data(
    results: list[SampleResult],
    task_labels: dict[str, str],
    sensitive: set[str],
    per_task: int = 1,
    max_chars: int = 160,
) -> list[dict[str, Any]]:
    """모델 간 출력이 갈린 샘플을 태스크별로 선택해 구조화 데이터로 반환.

    반환: [{task_id, label, sample_id, reference, sensitive, rows:[(model, output)]}]
    - (task, sample_id)로 그룹핑, 정규화 출력 고유값>1(모델 간 이견) 샘플만.
    - 고유값 큰 것부터 per_task개. sensitive(IMG-4)는 입력 없이 판정값만(D3).
    """
    grouped: dict[tuple[str, int], dict[str, tuple[str, Any]]] = {}
    prompts: dict[tuple[str, int], str] = {}
    for r in results:
        if r.finish_reason == "error":
            continue
        grouped.setdefault((r.task_id, r.sample_id), {})[r.model_id] = (r.model_output, r.reference)
        prompts.setdefault((r.task_id, r.sample_id), r.prompt)  # 입력(텍스트) — 모델 간 동일

    by_task: dict[str, list] = {}
    for (task_id, sid), model_out in grouped.items():
        if len(model_out) < 2:
            continue
        uniq = len({_norm(o) for o, _ in model_out.values()})
        if uniq < 2:
            continue
        by_task.setdefault(task_id, []).append((uniq, sid, model_out))

    slides: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        cands = sorted(by_task[task_id], key=lambda x: -x[0])[:per_task]
        is_image = task_id.startswith("IMG-")
        for uniq, sid, model_out in cands:
            ref = next(iter(model_out.values()))[1]
            rows = [
                (mid, _truncate(str(model_out[mid][0]).replace("\n", " "), max_chars))
                for mid in sorted(model_out)
            ]
            slides.append({
                "task_id": task_id,
                "label": task_labels.get(task_id, ""),
                "sample_id": sid,
                "reference": _truncate(str(ref), max_chars),
                "sensitive": task_id in sensitive,
                "is_image": is_image,
                # 입력: 텍스트 태스크는 prompt 전문(사람이 직접 판별용, 길이 넉넉히).
                #       이미지 태스크는 질문만 prompt에 있고 이미지는 image_data_uri로 별도 채움.
                "input_text": "" if is_image else _truncate(prompts.get((task_id, sid), ""), 1200),
                "image_data_uri": None,   # 이미지 태스크: generate_report에서 재로드해 채움(D3: NSFW 제외)
                "rows": rows,
            })
    return slides


def _attach_gallery_images(slides: list[dict[str, Any]], tasks_cfg_path: str = "config/tasks.yaml") -> None:
    """이미지 태스크 갤러리 샘플의 이미지를 재로드해 data URI로 채운다(in-place).

    갤러리에 뽑힌 소수 샘플만 처리. sensitive(NSFW)는 D3대로 이미지 미표시(건너뜀).
    재로드 실패(네트워크·데이터셋)는 조용히 건너뜀(이미지 없이 텍스트 비교는 여전히 유효).
    """
    image_slides = [g for g in slides if g.get("is_image") and not g.get("sensitive")]
    if not image_slides:
        return
    try:
        import yaml

        from src.adapters.images import pil_to_data_url
        from src.datasets_loader import load_registry
        from src.tasks.loader import discover_tasks

        registry = load_registry()
        task_classes = discover_tasks()
        cfg = yaml.safe_load(open(tasks_cfg_path, encoding="utf-8"))
        task_cfgs = {t["id"]: t for t in cfg.get("image_tasks", []) + cfg.get("text_tasks", [])}
        defaults = cfg.get("defaults", {})
        n, seed = defaults.get("samples", 50), defaults.get("seed", 42)

        # 태스크별로 한 번만 load_samples(같은 seed → 같은 subset), sample_id로 매칭
        by_task: dict[str, list] = {}
        for g in image_slides:
            by_task.setdefault(g["task_id"], []).append(g)

        for task_id, gs in by_task.items():
            cls = task_classes.get(task_id)
            if cls is None:
                continue
            inst = cls(task_cfgs.get(task_id, {}), registry)
            samples = inst.load_samples(n, seed)
            by_sid = {s.sample_id: s for s in samples}
            for g in gs:
                s = by_sid.get(g["sample_id"])
                img = s.inputs.get("image") if s else None
                if img is not None:
                    try:
                        g["image_data_uri"] = pil_to_data_url(img, max_side=512)
                    except Exception:
                        pass
    except Exception as e:
        print(f"  [갤러리 이미지 재로드 스킵] {type(e).__name__}: {e}")


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
            blocks.append(f"**샘플 #{g['sample_id']}**{tag} · 정답: `{g['reference']}`\n")
            # 입력 표시 (사람이 직접 판별용)
            if g.get("input_text"):
                inp = g["input_text"].strip().replace("\n", "\n> ")
                blocks.append(f"**입력:**\n> {inp}\n")
            elif g.get("image_data_uri") and reports_dir is not None:
                # data URI를 png로 저장해 링크(리포트에서 바로 보이게)
                fname = f"gallery_{task_id}_s{g['sample_id']}.png"
                try:
                    b64 = g["image_data_uri"].split(",", 1)[1]
                    (reports_dir / fname).write_bytes(base64.b64decode(b64))
                    blocks.append(f"**입력 이미지:**\n\n![{task_id} sample {g['sample_id']}]({fname})\n")
                except Exception:
                    pass
            elif g.get("is_image") and not g.get("sensitive"):
                blocks.append("_(입력 이미지 재로드 실패 — 아래 출력만 참고)_\n")
            blocks.append("| 모델 | 출력/판정 |")
            blocks.append("|---|---|")
            for mid, out in g["rows"]:
                blocks.append(f"| {mid} | {out.replace('|', chr(92) + '|')} |")
            blocks.append("")
    return "\n".join(blocks)
