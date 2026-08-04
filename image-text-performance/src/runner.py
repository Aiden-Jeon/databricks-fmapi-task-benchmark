"""벤치마크 실행 orchestrator (Phase 0 스캐폴딩).

실행 매트릭스 구성(모델×태스크×reasoning_mode), N/A 스킵 처리,
dry-run 지원, manifest 작성까지 구현.

실제 샘플 루프(dataset loading + task plugins)는 Phase 1에서 구현.

사용:
    python -m src.runner --config config/models.yaml --tasks config/tasks.yaml --dry-run
    python -m src.runner --config config/models.yaml --tasks config/tasks.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.adapters.fmapi import FMAPIClient
from src.config import load_models_config
from src.results import RunManifest, git_commit, make_run_id, write_manifest


def load_tasks_config(path: str | Path = "config/tasks.yaml") -> dict[str, Any]:
    """태스크 설정을 YAML에서 로드.

    구조: {image_tasks: [...], text_tasks: [...], defaults: {...}}
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_execution_matrix(
    models_cfg,
    tasks_cfg: dict[str, Any],
    models_filter: list[str] | None = None,
    reasoning_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """실행 매트릭스 구성: (모델 × 태스크 × reasoning_mode).

    이미지 태스크 중 vision 미지원 모델은 N/A로 마킹하고 스킵.

    Args:
        models_cfg: ModelsConfig 객체
        tasks_cfg: tasks.yaml dict
        models_filter: 포함할 모델 ID list (None이면 전부)
        reasoning_override: 사용할 reasoning_mode list (None이면 config에서)

    Returns:
        (실행 항목 list, N/A 스킵된 항목 수)
    """
    # 실제 모델·모드 결정
    models = models_cfg.models
    if models_filter:
        models = [m for m in models if m.id in models_filter]

    reasoning_modes = reasoning_override or models_cfg.reasoning_modes

    # 태스크 로드
    image_tasks = tasks_cfg.get("image_tasks", [])
    text_tasks = tasks_cfg.get("text_tasks", [])
    all_tasks = image_tasks + text_tasks

    # 매트릭스 구성
    matrix: list[dict[str, Any]] = []
    na_count = 0

    for model in models:
        for task in all_tasks:
            task_id = task["id"]
            is_image_task = task_id.startswith("IMG-")

            # N/A 검사: 이미지 태스크 + vision 미지원
            if is_image_task and not model.supports("vision"):
                na_count += len(reasoning_modes)  # 이 모델-태스크 조합의 모든 모드는 N/A
                continue

            for reasoning_mode in reasoning_modes:
                matrix.append(
                    {
                        "model_id": model.id,
                        "model_endpoint": model.endpoint,
                        "task_id": task_id,
                        "reasoning_mode": reasoning_mode,
                    }
                )

    return matrix, na_count


def main() -> int:
    """메인 runner 진입점."""
    parser = argparse.ArgumentParser(
        description="LLM 벤치마크 runner (Phase 0 스캐폴딩)",
    )
    parser.add_argument(
        "--config",
        default="config/models.yaml",
        help="모델·reasoning 설정 (default: config/models.yaml)",
    )
    parser.add_argument(
        "--tasks",
        default="config/tasks.yaml",
        help="태스크 설정 (default: config/tasks.yaml)",
    )
    parser.add_argument(
        "--models",
        help="포함할 모델 ID (쉼표 구분, e.g. 'opus,sol'). 생략 시 전부.",
    )
    parser.add_argument(
        "--reasoning-modes",
        help="사용할 reasoning mode (쉼표 구분, e.g. 'minimal,full'). 생략 시 config에서.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="샘플 수 제한 (빠른 테스트용, 생략 시 config 따름).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="매트릭스만 구성하고 FMAPI 호출 안함. manifest만 저장.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="결과 디렉터리 루트 (default: results)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="실행 전 기존 results/·reports/를 모두 삭제하고 새로 시작(index부터 재생성).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="per-task LLM-judge 채점 비활성(빠른 실행). 기본은 tasks.yaml의 metrics에 'judge'가 포함된 태스크에 judge를 실행한다.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="가장 최근 run 디렉터리를 이어서 실행. 이미 끝난 셀(scores.json에 존재)은 건너뛴다. --fresh와 함께 쓸 수 없음.",
    )

    args = parser.parse_args()

    # --fresh와 --resume은 상호 배타 (하나는 지우고 하나는 이어감)
    if args.fresh and args.resume:
        print("오류: --fresh와 --resume은 동시에 쓸 수 없습니다.", file=sys.stderr)
        return 1

    # --fresh: 기존 리포트·결과 전부 삭제 후 새로 시작 (파괴적 — 명시할 때만)
    if args.fresh and not args.dry_run:
        import shutil

        for d in (Path(args.out), Path("reports")):
            if d.exists():
                shutil.rmtree(d)
                print(f"[--fresh] {d}/ 삭제됨")

    # 설정 로드
    try:
        models_cfg = load_models_config(args.config)
        tasks_cfg = load_tasks_config(args.tasks)
    except FileNotFoundError as e:
        print(f"오류: 설정 파일을 찾을 수 없음: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"오류: 설정 파일 파싱 실패: {e}", file=sys.stderr)
        return 1

    # 필터 파싱
    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(",")]

    reasoning_override = None
    if args.reasoning_modes:
        reasoning_override = [m.strip() for m in args.reasoning_modes.split(",")]

    # 매트릭스 구성
    matrix, na_count = build_execution_matrix(
        models_cfg,
        tasks_cfg,
        models_filter=models_filter,
        reasoning_override=reasoning_override,
    )

    # 통계 계산
    total_cells = len(matrix)
    if total_cells == 0:
        print("경고: 실행할 항목이 없습니다.", file=sys.stderr)
        return 1

    # 모델별 셀 수 계산
    model_counts = {}
    for item in matrix:
        model_id = item["model_id"]
        model_counts[model_id] = model_counts.get(model_id, 0) + 1

    # Run ID 및 디렉터리 — resume면 가장 최근 run을 이어감, 아니면 새로 생성
    all_tasks = tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])
    task_ids = [t["id"] for t in all_tasks]

    resuming = args.resume and not args.dry_run
    if resuming:
        existing = sorted(
            (d for d in Path(args.out).glob("*") if d.is_dir()), key=lambda d: d.name
        )
        if not existing:
            print(f"오류: --resume이지만 {args.out}/에 이어갈 run이 없습니다.", file=sys.stderr)
            return 1
        run_dir = existing[-1]
        run_id = run_dir.name
        print(f"[--resume] 이어갈 run: {run_id}")
    else:
        run_id = make_run_id()
        run_dir = Path(args.out) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    model_list = []
    for model in models_cfg.models:
        if models_filter is None or model.id in models_filter:
            model_list.append(
                {
                    "id": model.id,
                    "endpoint": model.endpoint,
                    "family": model.family,
                }
            )

    # 재현성 메타(§12): 데이터셋 id·split, pricing 버전, 샘플·seed 스냅샷
    defaults = tasks_cfg.get("defaults", {})
    datasets_snapshot, pricing_snapshot = _reproducibility_meta(all_tasks)

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        models=model_list,
        reasoning_modes=reasoning_override or models_cfg.reasoning_modes,
        task_ids=task_ids,
        git_commit=git_commit(),
        datasets=datasets_snapshot,
        pricing=pricing_snapshot,
        samples_per_task=args.samples or defaults.get("samples", 50),
        seed=defaults.get("seed", 42),
        notes="dry-run" if args.dry_run else "full run",
    )

    # resume이면 기존 manifest 보존(덮어쓰지 않음)
    if resuming and (run_dir / "manifest.json").exists():
        manifest_path = run_dir / "manifest.json"
    else:
        manifest_path = write_manifest(run_dir, manifest)

    # 출력 (summary)
    print("=" * 70)
    print(f"실행 매트릭스 구성 완료")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"결과 디렉터리: {run_dir}")
    print()
    print(f"모델 수: {len(model_list)}")
    print(f"태스크 수: {len(task_ids)}")
    print(f"Reasoning 모드: {len(reasoning_override or models_cfg.reasoning_modes)}")
    print()
    print(f"총 실행 셀: {total_cells}")
    print(f"N/A 스킵(vision 미지원): {na_count}")
    print()
    print("모델별 셀 수:")
    for model_id, count in sorted(model_counts.items()):
        print(f"  {model_id}: {count}")
    print()
    print(f"Manifest 저장: {manifest_path}")
    print()

    if args.dry_run:
        print("DRY-RUN 모드: FMAPI 호출하지 않음.")
        print("=" * 70)
        return 0

    # 실제 실행 경로 (Phase 1)
    print("실제 샘플 루프 시작...")
    print("=" * 70)

    try:
        fmapi = FMAPIClient(
            profile=models_cfg.profile,
            timeout_seconds=models_cfg.runtime.timeout_seconds,
            max_retries=models_cfg.runtime.max_retries,
            backoff_initial_seconds=models_cfg.runtime.backoff_initial_seconds,
        )
    except Exception as e:
        print(f"오류: FMAPI 클라이언트 초기화 실패: {e}", file=sys.stderr)
        return 1

    try:
        rc = _run_samples(
            fmapi=fmapi,
            models_cfg=models_cfg,
            tasks_cfg=tasks_cfg,
            matrix=matrix,
            run_dir=run_dir,
            sample_cap=args.samples,
            enable_judge=not args.no_judge,
            resume=resuming,
        )
    finally:
        fmapi.close()

    print("=" * 70)
    return rc


def _run_samples(
    fmapi: FMAPIClient,
    models_cfg,
    tasks_cfg: dict[str, Any],
    matrix: list[dict[str, Any]],
    run_dir: Path,
    sample_cap: int | None,
    enable_judge: bool = True,
    resume: bool = False,
) -> int:
    """**태스크-메이저 스트리밍**으로 각 (모델×태스크×reasoning×샘플)을 호출·채점·저장.

    메모리 O(1)(샘플 수 무관) 설계:
    - 바깥 루프=태스크. 태스크당 샘플을 1회 로드(모델 간 공유), 태스크 끝나면 샘플·이미지 해제.
    - 샘플 결과는 전역 리스트에 쌓지 않고 셀마다 samples.jsonl에 **증분 append** 후 버림.
    - 채점은 태스크의 `make_accumulator()`(온라인 누적기)로 샘플당 add → 셀 끝에 finalize.
      누적기 미구현 태스크는 그 셀에 한해 버퍼링해 기존 score()로 폴백(점진 호환).
    - scores.json은 셀마다 원자적으로 갱신(크래시/resume 대비). resume면 이미 끝난 셀은 스킵.
    - 갤러리는 '현재 태스크'의 모델 간 상이 샘플만 소량 보관→태스크 끝에 top-1만 남기고 비움(O(태스크)).
    - 리포트는 맨 끝에 samples.jsonl을 1회 로드해 생성(실행 루프는 O(1) 유지).
    """
    from datetime import datetime, timezone

    from src.datasets_loader import load_registry
    from src.results import (
        SampleResult,
        append_sample_results,
        load_sample_results,
    )
    from src.tasks.loader import discover_tasks

    registry = load_registry()
    task_classes = discover_tasks()
    print(f"로드된 태스크: {sorted(task_classes)}")

    defaults = tasks_cfg.get("defaults", {})
    n_samples = sample_cap or defaults.get("samples", 50)
    seed = defaults.get("seed", 42)
    all_task_cfgs = {t["id"]: t for t in tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])}

    # matrix를 태스크-메이저로 재조직: task_id → [(model, mode), ...] (매트릭스 순서 유지)
    task_order: list[str] = []
    per_task_cells: dict[str, list[tuple[Any, str]]] = {}
    for cell in matrix:
        tid = cell["task_id"]
        if tid not in per_task_cells:
            per_task_cells[tid] = []
            task_order.append(tid)
        per_task_cells[tid].append((models_cfg.get_model(cell["model_id"]), cell["reasoning_mode"]))

    # 스트리밍 시작 전: 상태 준비. resume면 기존 scores/샘플 재사용, 아니면 새로.
    scores_path = run_dir / "scores.json"
    samples_path = run_dir / "samples.jsonl"
    if resume and scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        print(f"  [resume] 기존 scores.json에서 {len(scores)}개 셀 로드")
        # samples.jsonl을 scores.json과 정합화: 완료된 셀(scores에 존재)의 행만 남기고
        # 부분 기록(크래시 직전 셀은 scores 미기록)은 버린다. 안 그러면 그 셀 재실행 시
        # 옛 부분행 + 새 행이 중복돼 perf/비용 집계(_perf_by_model)가 부풀려진다.
        _reconcile_samples_with_scores(samples_path, set(scores.keys()))
    else:
        scores = {}
        # 새 실행이면 samples.jsonl 초기화(이전 잔여 제거)
        if samples_path.exists():
            samples_path.unlink()

    def _cell_done(key: str, want_judge: bool) -> bool:
        """resume 시 완료 판정: scores에 있고 error 없음. judge 태스크는 judge_mean까지 있어야 완료.

        judge가 실패해 judge_error만 남은 셀은 미완료로 보고 재실행(judge 재시도)한다.
        """
        e = scores.get(key)
        if not e or not isinstance(e.get("metrics"), dict) or "error" in e["metrics"]:
            return False
        if want_judge and enable_judge and "judge_mean" not in e["metrics"]:
            return False
        return True

    # judge 클라이언트(공유, 넉넉한 timeout). 실패해도 정량은 진행.
    judge_client = None
    if enable_judge:
        try:
            judge_client = FMAPIClient(
                profile=models_cfg.profile,
                timeout_seconds=max(60, models_cfg.runtime.timeout_seconds),
                max_retries=models_cfg.runtime.max_retries,
                backoff_initial_seconds=models_cfg.runtime.backoff_initial_seconds,
            )
        except Exception as e:
            print(f"  [judge 비활성 — 클라이언트 초기화 실패] {type(e).__name__}: {e}")

    executed = skipped_cells = resume_skipped = 0
    gallery_records: list[dict[str, Any]] = []
    gallery_path = run_dir / "gallery.jsonl"
    gallery_done_tasks: set[str] = set()  # 이미 gallery.jsonl에 기록된 task_id(resume 중복 방지)
    if resume and gallery_path.exists():
        # 기존 갤러리 유지 + 이미 기록된 task는 재기록하지 않음(중복 방지)
        with open(gallery_path, encoding="utf-8") as f:
            for line in f:
                try:
                    gallery_done_tasks.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    elif gallery_path.exists():
        gallery_path.unlink()

    try:
        for task_id in task_order:
            cls = task_classes.get(task_id)
            cfg = all_task_cfgs.get(task_id, {})
            if cls is None:
                skipped_cells += len(per_task_cells[task_id])
                continue
            inst = cls(cfg, registry)
            try:
                samples = inst.load_samples(n_samples, seed)
            except Exception as e:
                print(f"  [샘플 로드 실패] {task_id}: {type(e).__name__}: {e}")
                skipped_cells += len(per_task_cells[task_id])
                continue
            if not samples:
                skipped_cells += len(per_task_cells[task_id])
                continue

            # 이미지 태스크: 갤러리용 썸네일을 지금 저장(비민감만). 원본(풀해상도) PIL 이미지는
            # build_prompt가 모델마다 필요로 하므로 태스크 동안 유지하고, 태스크 끝의 `del samples`로 회수.
            if getattr(inst, "is_vision", False) and not getattr(inst, "sensitive", False):
                _save_sample_images(run_dir, task_id, samples)

            want_judge = "judge" in (cfg.get("metrics") or [])
            is_image = task_id.startswith("IMG-")
            sensitive = bool(getattr(inst, "sensitive", False))
            # 현재 태스크 한정 갤러리 버퍼: {sample_id: {model_id: short_output}}
            per_task_outputs: dict[int, dict[str, str]] = {}
            ref_by_sid: dict[int, Any] = {s.sample_id: s.reference for s in samples}

            for model, mode in per_task_cells[task_id]:
                key = f"{model.id}::{task_id}::{mode}"
                if resume and _cell_done(key, want_judge):
                    resume_skipped += 1
                    print(f"  [resume-skip] {key}")
                    continue

                params = model.reasoning_params(mode)
                acc = inst.make_accumulator() if hasattr(inst, "make_accumulator") else None
                buf_parsed: list[Any] | None = None if acc is not None else []
                buf_samples: list[Any] | None = None if acc is not None else []
                # judge를 이 셀에서 돌릴지 여부(태스크가 judge_scores로 자체 집계하므로 별도 누적기 불필요).
                do_judge = (
                    enable_judge and judge_client is not None
                    and want_judge and hasattr(inst, "judge_scores")
                )
                cell_results: list[SampleResult] = []
                print(f"  [{model.id}/{task_id}/{mode}] {len(samples)}샘플 실행...", flush=True)

                for s in samples:
                    messages = inst.build_prompt(s)
                    t0 = time.perf_counter()
                    try:
                        resp = fmapi.chat(
                            model.endpoint, messages,
                            max_tokens=models_cfg.runtime.max_tokens, extra_params=params,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        output_text = resp.text
                        req_id, finish, usage = resp.request_id, resp.finish_reason, resp.usage
                    except Exception as e:
                        latency_ms = (time.perf_counter() - t0) * 1000
                        output_text = f"__ERROR__: {type(e).__name__}: {e}"
                        req_id, finish, usage = None, "error", {}

                    try:
                        parsed = inst.parse_output(output_text, s)
                    except Exception:
                        parsed = None

                    if acc is not None:
                        try:
                            acc.add(parsed, s)
                        except Exception as e:
                            # add 실패는 드물지만 조용히 삼키면 n_evaluated가 어긋난다 → 경고 출력.
                            print(f"    [acc.add 실패] {key} s{s.sample_id}: {type(e).__name__}: {e}")
                    else:
                        buf_parsed.append(parsed)
                        buf_samples.append(s)

                    cell_results.append(SampleResult(
                        model_id=model.id, task_id=task_id, sample_id=s.sample_id,
                        reasoning_mode=mode, prompt=_truncate(_prompt_text(messages), 2000),
                        model_output=output_text, reference=s.reference, request_id=req_id,
                        finish_reason=finish, usage=usage, latency_ms_local=latency_ms,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                    # 갤러리 버퍼(짧게만): 모델 간 상이 판정용
                    if finish != "error":
                        per_task_outputs.setdefault(s.sample_id, {})[model.id] = _truncate(
                            str(output_text).replace("\n", " "), 200
                        )
                    executed += 1

                # 셀 완료: 결과 증분 저장(크래시 대비) 후 버림
                append_sample_results(run_dir, cell_results)
                del cell_results

                # 채점 finalize
                try:
                    metrics = acc.finalize() if acc is not None else inst.score(buf_parsed, buf_samples)
                except Exception as e:
                    metrics = {"error": f"{type(e).__name__}: {e}"}

                # per-task judge (config-gated). 정량 정상일 때만.
                # judge_scores()는 parsed 리스트 전체를 요구하므로: 폴백 경로는 buf_parsed를,
                # 누적기 경로는 방금 저장한 samples.jsonl에서 parsed를 재구성해 넘긴다(한 셀 범위, 작음).
                if do_judge and isinstance(metrics, dict) and "error" not in metrics:
                    print(f"  [judge] {model.id}/{task_id}/{mode}: {len(samples)}샘플 채점...", flush=True)
                    try:
                        if acc is None:
                            jr = inst.judge_scores(buf_parsed, buf_samples, judge_client, models_cfg.judge)
                        else:
                            jr = _run_judge_streaming(
                                inst, samples, run_dir, model.id, mode, judge_client, models_cfg.judge
                            )
                        metrics.update(_normalize_judge(jr))
                    except Exception as e:
                        metrics["judge_error"] = f"{type(e).__name__}: {e}"

                scores[key] = {
                    "model_id": model.id, "task_id": task_id, "reasoning_mode": mode,
                    "n": len(samples), "metrics": metrics,
                }
                _atomic_write_json(scores_path, scores)  # 셀마다 체크포인트

            # 태스크 종료: 갤러리 top-1(모델 간 최다 상이) 기록 후 버퍼 비움.
            # resume에서 이미 기록된 task는 건너뜀(중복 방지). 이번에 셀을 하나도 안 돌린
            # (전부 resume-skip) 태스크는 per_task_outputs가 비어 _record_gallery가 no-op.
            if task_id not in gallery_done_tasks:
                _record_gallery(gallery_path, gallery_records, task_id, is_image, sensitive,
                                 per_task_outputs, ref_by_sid)
            per_task_outputs.clear()
            # 샘플(이미지 포함) 전부 해제 → 다음 태스크 전에 메모리 회수
            del samples, ref_by_sid, inst
            import gc
            gc.collect()
    finally:
        if judge_client is not None:
            judge_client.close()

    print(f"\n실행 완료: {executed}개 샘플 호출"
          + (f", resume 스킵 {resume_skipped}셀" if resume_skipped else "")
          + (f", 미구현/로드실패 스킵 {skipped_cells}셀" if skipped_cells else ""))
    print(f"결과 저장: {samples_path}")

    # 리포트 생성: 실행 루프 밖에서 samples.jsonl을 1회 로드(여기서만 전체 메모리 사용).
    try:
        from src.report.generate import generate_report

        results_for_report = load_sample_results(run_dir)
        report_path = generate_report(run_dir, results_for_report, scores, models_cfg)
        print(f"리포트 생성: {report_path}")
    except Exception as e:
        print(f"  [리포트 생성 스킵] {type(e).__name__}: {e}")

    try:
        from src.report.index import rebuild_index

        idx = rebuild_index()
        print(f"인덱스 갱신: {idx}")
    except Exception as e:
        print(f"  [인덱스 갱신 스킵] {type(e).__name__}: {e}")

    return 0


def _run_judge_streaming(inst, samples, run_dir, model_id, mode, judge_client, judge_endpoint):
    """누적기 경로에서 per-task judge를 수행.

    judge_scores()는 parsed 리스트를 요구하지만, 스트리밍에선 parsed를 보관하지 않으므로
    이 시점에 samples.jsonl에서 해당 셀의 model_output을 읽어 parsed를 재구성한다(디스크 기반, O(샘플)이지만
    한 셀 범위라 작음). 태스크의 judge_scores 계약을 그대로 사용한다.
    """
    from src.results import load_sample_results

    rows = [
        r for r in load_sample_results(run_dir)
        if r.model_id == model_id and r.task_id == inst.task_id and r.reasoning_mode == mode
    ]
    by_sid = {r.sample_id: r for r in rows}
    parsed = []
    for s in samples:
        r = by_sid.get(s.sample_id)
        out = r.model_output if r else ""
        try:
            parsed.append(inst.parse_output(out, s))
        except Exception:
            parsed.append(None)
    return inst.judge_scores(parsed, samples, judge_client, judge_endpoint)


def _atomic_write_json(path: Path, obj: Any) -> None:
    """tmp에 쓰고 os.replace로 교체(부분 기록 방지). scores.json 체크포인트용."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _reconcile_samples_with_scores(samples_path: Path, done_keys: set[str]) -> None:
    """resume 시 samples.jsonl을 scores.json과 정합화한다.

    완료된 셀(scores에 있는 key = "model::task::mode")의 행만 남기고, scores에 없는
    셀의 행(크래시 직전 부분 기록)은 버린다. 그래야 그 셀을 재실행해 다시 append해도
    옛 부분행과 중복되지 않아 perf·비용 집계가 정확하다. 원자적 교체로 안전하게 재작성.
    파일이 없으면 no-op.
    """
    import os

    if not samples_path.exists():
        return
    kept: list[str] = []
    dropped = 0
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                key = f"{d['model_id']}::{d['task_id']}::{d['reasoning_mode']}"
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            if key in done_keys:
                kept.append(line)
            else:
                dropped += 1
    tmp = samples_path.with_suffix(samples_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
        f.flush()
    os.replace(tmp, samples_path)
    if dropped:
        print(f"  [resume] samples.jsonl 정합화: 미완료 셀 {dropped}행 제거, {len(kept)}행 유지")


def _record_gallery(gallery_path, records, task_id, is_image, sensitive, per_task_outputs, ref_by_sid):
    """현재 태스크에서 모델 간 출력이 가장 갈린 샘플 1개를 갤러리 레코드로 남긴다(O(1) 유지).

    per_task_outputs: {sample_id: {model_id: short_output}}. 고유 출력 수>1인 샘플 중 최다를 선택.
    """
    best_sid, best_uniq = None, 1
    for sid, mo in per_task_outputs.items():
        if len(mo) < 2:
            continue
        uniq = len({v for v in mo.values()})
        if uniq > best_uniq:
            best_uniq, best_sid = uniq, sid
    if best_sid is None:
        return
    rec = {
        "task_id": task_id, "sample_id": best_sid, "is_image": is_image, "sensitive": sensitive,
        "reference": _truncate(str(ref_by_sid.get(best_sid, "")), 200),
        "rows": sorted(per_task_outputs[best_sid].items()),
    }
    records.append(rec)
    with open(gallery_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _reproducibility_meta(all_tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """manifest용 데이터셋·pricing 스냅샷 (§12 재현성). 로드 실패해도 빈 dict."""
    datasets_snapshot: dict[str, Any] = {}
    try:
        from src.datasets_loader import load_registry

        registry = load_registry()
        for t in all_tasks:
            for lang, key in (t.get("datasets") or {}).items():
                entry = registry.get(key, {})
                datasets_snapshot[key] = {
                    "hf_id": entry.get("hf_id"),
                    "split": entry.get("split"),
                    "config": entry.get("config"),
                }
    except Exception:
        pass

    pricing_snapshot: dict[str, Any] = {}
    try:
        from src.cost.pricing import load_pricing

        p = load_pricing()
        pricing_snapshot = {"usd_per_dbu": p.get("usd_per_dbu"), "routing": p.get("routing")}
    except Exception:
        pass

    return datasets_snapshot, pricing_snapshot


def _save_sample_images(run_dir: Path, task_id: str, samples: list) -> None:
    """이미지 태스크 샘플의 입력 이미지를 run_dir/images/에 저장(갤러리 정합성).

    파일명: <task_id>_s<sample_id>.jpg. 실행 시점 저장이라 sample_id ↔ 이미지가 정확.
    실패는 조용히 무시(이미지 없이 텍스트 비교는 여전히 유효).
    """
    img_dir = run_dir / "images"
    try:
        from PIL import Image

        img_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            img = s.inputs.get("image") if isinstance(s.inputs, dict) else None
            if img is None or not isinstance(img, Image.Image):
                continue
            im = img.convert("RGB") if img.mode not in ("RGB", "L") else img
            w, h = im.size
            if max(w, h) > 512:
                sc = 512 / max(w, h)
                im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))))
            im.save(img_dir / f"{task_id}_s{s.sample_id}.jpg", "JPEG", quality=85)
    except Exception as e:
        print(f"  [이미지 저장 스킵] {task_id}: {type(e).__name__}: {e}")


def _normalize_judge(jr: dict[str, Any]) -> dict[str, Any]:
    """태스크별로 제각각인 judge_scores() 반환을 표준화한다.

    두 계열의 키를 흡수: {judge_mean, n_judged}(txt_1/2/4)와
    {judge_score_mean, n_evaluated, per_language}(txt_5/img_1).
    반환: {judge_mean: float}(대표 수치, 1–5 스케일) + {judge_detail: {...}}(감사용, 비수치).
    judge_mean만 수치 top-level이라 정량표에 한 열로 노출된다.
    """
    if not isinstance(jr, dict):
        return {}
    mean = jr.get("judge_mean")
    if mean is None:
        mean = jr.get("judge_score_mean")
    n = jr.get("n_judged")
    if n is None:
        n = jr.get("n_evaluated")
    out: dict[str, Any] = {}
    if isinstance(mean, (int, float)):
        out["judge_mean"] = round(float(mean), 4)
    detail: dict[str, Any] = {"n": n, "scores": jr.get("judge_scores")}
    if jr.get("per_language"):
        detail["per_language"] = jr["per_language"]
    out["judge_detail"] = detail
    return out


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    """저장·표시용으로 messages에서 텍스트만 추출."""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


if __name__ == "__main__":
    sys.exit(main())
