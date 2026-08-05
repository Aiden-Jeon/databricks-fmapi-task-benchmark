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
from src.config import ConfigValidationError, load_models_config, validate_models_config
from src.results import RunManifest, git_commit, make_run_id, write_manifest
from src.scoring.accumulators import CALL_FAILED

# 한 셀에서 이 비율을 넘게 호출이 실패하면 그 셀의 점수는 모델 성능이 아니라 장애의 산물로 본다.
# 실패 응답은 파싱 실패로 0점 채점되지만 예외가 아니라 metrics에 'error'가 없어, 그냥 두면
# 리포트에 정상 수치처럼 실린다(실측: IMG-6 opus 19/30 실패 → cell_f1 0.290 vs 실제 0.841).
# `unreliable` 표시 + resume 시 재실행 대상(_poisoned_cells) 판정에 같은 값을 쓴다.
UNRELIABLE_FAILURE_RATE = 0.5

# 전체 호출 중 이 비율을 넘게 실패하면 실행 자체를 실패로 보고 **non-zero exit**를 낸다.
# 실패해도 exit 0이면 자동화(CI·스크립트·에이전트)가 "성공"으로 오판한다 — 모델 전체가
# 403/400이어도 "실행 완료"가 찍히던 문제. 개별 셀의 산발적 실패(임계 미만)는 채점에서
# 제외되고 리포트에 드러나므로 0을 유지한다. 10%는 실측 기준(정상 run은 0~2%)에서 잡았다.
GLOBAL_FAILURE_EXIT_RATE = 0.10

# judge 실패율이 이 값을 넘으면 실행을 실패로 본다. judge가 대량 실패하면 생성 태스크의
# 대표 수치(judge_mean)가 사실상 비어 있는데도 예전엔 exit 0이었다 — 실측: max_tokens
# 부족으로 IMG-1 judge가 30/30 실패했는데 "실행 완료"로 끝났다. 산발적 잘림(요약 태스크에서
# 30건당 1~2건)은 정상 범위라 5%로 둔다.
JUDGE_FAILURE_EXIT_RATE = 0.05

# samples.jsonl에 몇 샘플마다 flush할지. 셀 전체(보통 30개)를 다 돌고 나서 한 번에 쓰면
# 중단 시 이미 과금된 호출 결과가 통째로 사라진다. 5개마다 append하면 손실이 최대 4개로
# 줄고, I/O도 샘플당 수 초 걸리는 호출에 비하면 무시할 수준이다.
SAMPLE_FLUSH_EVERY = 5


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
        "--profile",
        default=None,
        help="Databricks CLI 프로파일. 생략 시 config/models.yaml의 profile을 쓴다. "
             "어느 워크스페이스로 호출·과금되는지 실행마다 명시하고 싶을 때 사용.",
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

    # ⚠️ --fresh 삭제는 **여기서 하지 않는다**. 설정 검증·매트릭스 구성·클라이언트 초기화가
    # 모두 성공한 뒤(_purge_previous_runs 호출 지점)로 미룬다.
    # 실측 사고(2026-08-06): 삭제가 검증보다 앞서 있어서, 실행이 첫 셀에서 중단됐을 때
    # 직전 리포트가 이미 사라져 README의 '최신 리포트' 링크가 깨졌다(git에서 복구했다).
    # 설정 오류·인증 실패로 한 셀도 못 돌리는 경우가 흔하므로, 그때는 기존 결과를 지키는 게 맞다.

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

    # ── 설정 교차 검증 (모델 추가 시 1차 방어선) ──────────────────────
    # 새 모델을 붙일 때 조용히 잘못 도는 경우가 많다: reasoning 모드 누락(기본값=ON일 수
    # 있는데 리포트는 OFF로 표기), capabilities 오타(태스크 전체 N/A), ID 중복(모델 소실),
    # pricing 미등록(비용 0 → "가장 저렴"으로 오선정). 실행 전에 잡아 명확히 실패시킨다.
    try:
        cfg_for_check = models_cfg
        if reasoning_override:
            # 실제로 돌릴 모드만 검증 대상으로 삼는다(--reasoning-modes로 좁힌 경우).
            cfg_for_check = models_cfg.model_copy(update={"reasoning_modes": reasoning_override})
        warns = validate_models_config(cfg_for_check)
        for w in warns:
            print(f"  ⚠️  [설정 경고] {w}")
    except ConfigValidationError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

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
        # results_root를 넘겨 기존 디렉터리와 충돌하지 않는 ID를 받는다(같은 분에 두 실행 시).
        run_id = make_run_id(results_root=args.out)
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
        profile=args.profile or models_cfg.profile,
        notes="dry-run" if args.dry_run else "full run",
    )

    # resume이면 기존 manifest를 **보존하되 구성 변화가 있으면 거부**한다.
    # 예전엔 무조건 보존해서, 모델을 추가한 뒤 --resume하면 기존 3모델 run에 새 모델 결과가
    # 섞이는데 manifest에는 옛 3모델만 남았다 → "이 리포트는 어떤 구성으로 뽑혔나"가 거짓이 되고
    # 시점 비교가 무의미해진다. 모델·태스크·reasoning 모드·샘플 수가 달라지면 **새 run**을 써야 한다.
    manifest_path = run_dir / "manifest.json"
    if resuming and manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        drift = _manifest_drift(prev, manifest)
        if drift:
            print(
                "\n오류: --resume 대상 run과 현재 설정이 다릅니다. 이어서 돌리면 서로 다른 "
                "구성의 결과가 한 run에 섞이고 manifest가 사실과 달라집니다:",
                file=sys.stderr,
            )
            for d in drift:
                print(f"   - {d}", file=sys.stderr)
            print(
                "\n해결: (a) 구성을 원래대로 돌려 resume하거나, (b) --fresh 또는 resume 없이 "
                "**새 run**으로 실행하세요(권장 — 시점 비교가 유효해집니다).",
                file=sys.stderr,
            )
            return 1
        # 구성이 같으면 기존 manifest를 그대로 둔다(created_at·git_commit 등 시점 정보 보존).
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
    # 모델별 실효 런타임을 찍어 둔다 — 오버라이드가 실제로 적용됐는지 로그로 확인 가능해야
    # 한다(공통값이 느린 모델에 맞지 않아 타임아웃 실패가 나던 문제 때문에 도입).
    print("모델별 실효 런타임(timeout/재시도/max_tokens):")
    for m in models_cfg.models:
        if models_filter and m.id not in models_filter:
            continue
        rt = m.effective_runtime(models_cfg.runtime)
        tag = " (모델별 오버라이드)" if m.runtime is not None else ""
        print(f"  {m.id}: {rt.timeout_seconds}s / {rt.max_retries}회 / {rt.max_tokens} tokens{tag}")
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

    # 프로파일은 여기서 한 번 확정하고 아래로 전달한다(모델 호출·judge가 같은 값을 쓰도록).
    # --profile이 config를 덮어쓴다. 어느 워크스페이스로 호출·과금되는지 실행 로그에 남긴다 —
    # 잘못된 프로파일은 엉뚱한 워크스페이스에 과금되고 IP ACL 403 같은 실패의 원인도 흐린다.
    profile = args.profile or models_cfg.profile
    print(f"Databricks 프로파일: {profile}"
          + (" (--profile로 지정)" if args.profile else " (config/models.yaml 기본값)"))

    try:
        fmapi = FMAPIClient(
            profile=profile,
            timeout_seconds=models_cfg.runtime.timeout_seconds,
            max_retries=models_cfg.runtime.max_retries,
            backoff_initial_seconds=models_cfg.runtime.backoff_initial_seconds,
        )
    except Exception as e:
        print(f"오류: FMAPI 클라이언트 초기화 실패: {e}", file=sys.stderr)
        return 1

    # --fresh 삭제는 **여기서** 한다: 설정 검증·매트릭스 구성·클라이언트 초기화가 모두
    # 통과해 실제로 호출을 시작할 수 있음이 확인된 뒤. 그 전에 지우면 설정 오류·인증 실패로
    # 한 셀도 못 돌렸을 때 직전 리포트만 잃는다(실측 사고 — 위 주석 참고).
    # 게다가 삭제가 아니라 **보관 이동**이라, 새 run이 중간에 죽어도 이전 결과를 되찾을 수 있다.
    if args.fresh:
        _purge_previous_runs(Path(args.out), Path("reports"), keep_run_id=run_id)

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
            profile=profile,
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
    profile: str | None = None,
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
        # IP 차단·대량 장애로 셀의 모델 응답이 대부분 __ERROR__가 되면, 각 태스크의 score()는
        # 예외 없이 0.0류 hollow 지표를 산출해 scores.json에 'error' 키 없이 기록된다.
        # 그러면 _cell_done이 '완료'로 오판해 스킵한다(옛 버그). 그런 셀은 scores에서 제거해
        # 미완료로 되돌리고(→ 재실행), 아래 정합화가 옛 에러 행을 samples.jsonl에서 걷어낸다.
        poisoned = {k for k in _poisoned_cells(samples_path) if k in scores}
        if poisoned:
            for k in sorted(poisoned):
                del scores[k]
            print(f"  [resume] 응답 대량 실패(>50% __ERROR__)로 재실행할 셀 {len(poisoned)}개: "
                  f"{', '.join(sorted(poisoned))}")
        # samples.jsonl을 scores.json과 정합화: 완료된 셀(scores에 존재)의 행만 남기고
        # 부분 기록(크래시 직전 셀은 scores 미기록)·오염 셀 행은 버린다. 안 그러면 그 셀
        # 재실행 시 옛 행 + 새 행이 중복돼 perf/비용 집계(_perf_by_model)가 부풀려진다.
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
                # 호출부에서 확정한 프로파일을 쓴다(--profile 반영). 없으면 config 기본값.
                profile=profile or models_cfg.profile,
                timeout_seconds=max(60, models_cfg.runtime.timeout_seconds),
                max_retries=models_cfg.runtime.max_retries,
                backoff_initial_seconds=models_cfg.runtime.backoff_initial_seconds,
            )
        except Exception as e:
            print(f"  [judge 비활성 — 클라이언트 초기화 실패] {type(e).__name__}: {e}")

    executed = skipped_cells = resume_skipped = 0
    # 요청보다 적게 로드된 태스크: {task_id: (로드됨, 요청됨)}. 종료 요약·셀 메타에 쓴다.
    shortfall_tasks: dict[str, tuple[int, int]] = {}
    # 아예 실행하지 못한 태스크: {task_id: 이유}. **종료 코드를 실패로 만든다** —
    # 태스크가 통째로 누락됐는데 "실행 완료 exit 0"이 되던 문제를 막는다.
    failed_tasks: dict[str, str] = {}
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
            # 아래 세 경로는 **모두 실행 실패**다(의도된 N/A 스킵과 다르다) — 종료 코드에
            # 반영해야 한다. 예전엔 skipped_cells로 세고 그 수를 기대 셀 수에서 빼서,
            # 태스크가 통째로 누락돼도 exit 0이 나왔다. failed_tasks에 이유와 함께 남긴다.
            if cls is None:
                print(f"  [태스크 미구현] {task_id}: 플러그인이 없습니다")
                failed_tasks[task_id] = "플러그인 미구현"
                skipped_cells += len(per_task_cells[task_id])
                continue
            inst = cls(cfg, registry)
            try:
                samples = inst.load_samples(n_samples, seed)
            except Exception as e:
                print(f"  [샘플 로드 실패] {task_id}: {type(e).__name__}: {e}")
                failed_tasks[task_id] = f"샘플 로드 실패: {type(e).__name__}: {str(e)[:120]}"
                skipped_cells += len(per_task_cells[task_id])
                continue
            if not samples:
                print(f"  [샘플 0개] {task_id}: 데이터셋이 샘플을 주지 않았습니다")
                failed_tasks[task_id] = "샘플 0개"
                skipped_cells += len(per_task_cells[task_id])
                continue

            # 요청보다 적게 로드되면 **눈에 보이게** 남긴다. `--samples 30`은 최대 요청값일 뿐이라
            # 태스크가 10개만 로드해도 예전엔 조용히 통과했고(실측: TXT-2가 미러 split 문제로
            # 항상 n=10), 리포트에는 다른 태스크와 같은 무게로 실렸다. 표본이 작아진 사실이
            # scores.json·리포트에 드러나야 수치를 올바르게 해석할 수 있다.
            if len(samples) < n_samples:
                shortfall_tasks[task_id] = (len(samples), n_samples)
                print(f"  ⚠️  [{task_id}] 요청 {n_samples}개 중 {len(samples)}개만 로드됨 "
                      f"— 이 태스크의 표본이 작다(리포트에 n_requested로 표기)")

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
                # 모델별 런타임(있으면) 적용 — 느린 모델·긴 출력 모델이 공통값 때문에
                # 타임아웃 실패를 내고 그 실패가 성능으로 오해되는 것을 막는다.
                rt = model.effective_runtime(models_cfg.runtime)
                acc = inst.make_accumulator() if hasattr(inst, "make_accumulator") else None
                buf_parsed: list[Any] | None = None if acc is not None else []
                buf_samples: list[Any] | None = None if acc is not None else []
                # judge를 이 셀에서 돌릴지 여부(태스크가 judge_scores로 자체 집계하므로 별도 누적기 불필요).
                do_judge = (
                    enable_judge and judge_client is not None
                    and want_judge and hasattr(inst, "judge_scores")
                )
                cell_results: list[SampleResult] = []
                n_call_failed = 0   # 이 셀의 호출 실패 수 → 점수와 함께 기록(아래 주석 참고)
                print(f"  [{model.id}/{task_id}/{mode}] {len(samples)}샘플 실행...", flush=True)

                for s in samples:
                    messages = inst.build_prompt(s)
                    t0 = time.perf_counter()
                    try:
                        resp = fmapi.chat(
                            model.endpoint, messages,
                            max_tokens=rt.max_tokens, extra_params=params,
                            timeout_seconds=rt.timeout_seconds, max_retries=rt.max_retries,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000
                        output_text = resp.text
                        req_id, finish, usage = resp.request_id, resp.finish_reason, resp.usage
                    except Exception as e:
                        latency_ms = (time.perf_counter() - t0) * 1000
                        output_text = f"__ERROR__: {type(e).__name__}: {e}"
                        req_id, finish, usage = None, "error", {}

                    # 호출 실패와 파싱 실패를 **다른 값으로** 구분한다:
                    # - CALL_FAILED: 응답 자체를 못 받음(502·타임아웃) → 인프라 문제 → 채점 제외
                    # - None: 응답은 받았지만 형식을 못 맞춤 → **능력 문제 → 0점 채점**
                    # 둘을 None으로 합치면(2026-08-05~08-06 상태) 형식을 대부분 못 맞추는
                    # 새 모델이 "성공한 일부"만으로 높은 점수를 받는다. 반대로 둘 다 0점으로
                    # 채점하면 엔드포인트 장애가 성능 저하로 보인다(실측: opus 502 11/30 →
                    # micro_f1 0.671 vs 성공분만 0.786). 그래서 두 축을 분리해 둔다.
                    if finish == "error":
                        parsed = CALL_FAILED
                    else:
                        try:
                            parsed = inst.parse_output(output_text, s)
                        except Exception:
                            parsed = None    # 파싱 예외 = 형식 불일치 → 0점

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
                        # 가운데 생략으로 head+tail 보존 → 표/문서 뒤에 오는 질문이 안 잘림(갤러리 질문 표시용).
                        reasoning_mode=mode, prompt=_elide_middle(_prompt_text(messages), 2400),
                        model_output=output_text, reference=s.reference, request_id=req_id,
                        finish_reason=finish, usage=usage, latency_ms_local=latency_ms,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                    # 갤러리 버퍼(짧게만): 모델 간 상이 판정용
                    if finish != "error":
                        per_task_outputs.setdefault(s.sample_id, {})[model.id] = _truncate(
                            str(output_text).replace("\n", " "), 200
                        )
                    else:
                        n_call_failed += 1
                    executed += 1

                    # **샘플 단위 flush**: 버퍼가 차면 즉시 samples.jsonl에 append한다.
                    # 예전엔 셀의 30개 호출과 judge가 다 끝난 뒤에야 기록해서, 중단되면
                    # 이미 돈을 쓴 호출 결과가 통째로 사라지고 재실행 시 전부 다시 호출했다.
                    # (scores.json은 여전히 셀 단위 — 셀 점수는 전체 샘플이 있어야 확정된다.
                    #  다만 samples.jsonl이 남으면 어디까지 호출했는지 사후 확인은 가능하다.)
                    if len(cell_results) >= SAMPLE_FLUSH_EVERY:
                        append_sample_results(run_dir, cell_results)
                        cell_results.clear()

                # 남은 버퍼 flush (셀 완료)
                if cell_results:
                    append_sample_results(run_dir, cell_results)
                    cell_results.clear()
                del cell_results

                # 채점 finalize
                try:
                    if acc is not None:
                        metrics = acc.finalize()
                    else:
                        # 폴백 score() 경로: 태스크의 score()는 CALL_FAILED sentinel을 모르므로
                        # **호출 실패 샘플을 여기서 제거**해서 넘긴다(파싱 실패 None은 그대로
                        # 넘겨 0점으로 채점되게 한다 — 능력 문제이므로).
                        keep = [
                            (p, s) for p, s in zip(buf_parsed, buf_samples) if p is not CALL_FAILED
                        ]
                        n_dropped = len(buf_parsed) - len(keep)
                        metrics = inst.score([p for p, _ in keep], [s for _, s in keep])
                        if n_dropped and isinstance(metrics, dict):
                            metrics["n_skipped"] = n_dropped
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

                # 호출 실패 수를 점수와 **같은 자리에** 남긴다. 실패 응답("__ERROR__: ...")은
                # 파싱이 실패해 0점으로 채점되는데, 예외가 아니라서 metrics에 'error' 키가 없다
                # → 리포트에 정상 수치처럼 실린다. 실측 사고: IMG-6가 15s 타임아웃으로 opus
                # 19/30 실패했는데 cell_f1 0.290이 "성능 낮음"으로 읽혔다(실제는 0.841).
                # n_call_failed / call_failure_rate로 리포트가 신뢰도를 판단할 수 있게 한다.
                entry: dict[str, Any] = {
                    "model_id": model.id, "task_id": task_id, "reasoning_mode": mode,
                    "n": len(samples), "metrics": metrics,
                    # 요청 샘플 수. n < n_requested면 표본이 작다는 뜻이고 리포트가 그걸 표기한다.
                    "n_requested": n_samples,
                }
                if n_call_failed:
                    entry["n_call_failed"] = n_call_failed
                    entry["call_failure_rate"] = round(n_call_failed / max(1, len(samples)), 4)
                    if n_call_failed / max(1, len(samples)) > UNRELIABLE_FAILURE_RATE:
                        # 절반 이상 실패한 셀의 점수는 모델 성능이 아니라 장애의 산물이다.
                        entry["unreliable"] = True
                        print(f"    ⚠️  [{key}] 호출 {n_call_failed}/{len(samples)} 실패 "
                              f"→ 점수 신뢰불가로 표시(리포트에 경고 표기)")
                    else:
                        print(f"    [{key}] 호출 실패 {n_call_failed}/{len(samples)}")
                scores[key] = entry
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

    if shortfall_tasks:
        print(f"\n⚠️  요청보다 적게 로드된 태스크 {len(shortfall_tasks)}개 (표본이 작다):")
        for t, (got, want) in sorted(shortfall_tasks.items()):
            print(f"   {t}: {got}/{want}")
        print("   → 데이터셋 split·행 수를 확인할 것(`.info.splits`). 리포트에는 n으로 표기된다.")

    # 실패 요약을 종료 직전에 한 번 더 모아 보여준다. 셀별 경고는 긴 로그에 묻히기 쉬운데,
    # 실패를 못 보고 리포트를 그대로 신뢰하는 것이 이 벤치마크의 반복된 사고 원인이었다.
    failed_cells = {k: v for k, v in scores.items() if v.get("n_call_failed")}
    if failed_cells:
        unreliable = [k for k, v in failed_cells.items() if v.get("unreliable")]
        print(f"\n⚠️  호출 실패가 있는 셀 {len(failed_cells)}개:")
        for k in sorted(failed_cells):
            v = failed_cells[k]
            mark = " ← 신뢰불가(순위 제외)" if v.get("unreliable") else ""
            print(f"   {k}: {v['n_call_failed']}/{v['n']} "
                  f"({v['call_failure_rate']:.1%}){mark}")
        if unreliable:
            print(f"   → 신뢰불가 {len(unreliable)}셀은 리포트 순위·요약에서 제외된다. "
                  f"timeout_seconds 상향 후 --resume으로 재실행 권장.")
    judge_failed = {
        k: (v["metrics"].get("judge_detail") or {}).get("n_failed")
        for k, v in scores.items()
        if isinstance(v.get("metrics"), dict)
        and (v["metrics"].get("judge_detail") or {}).get("n_failed")
    }
    if judge_failed:
        print(f"\n⚠️  judge 파싱 실패가 있는 셀 {len(judge_failed)}개 "
              f"(해당 샘플은 평균에서 제외됨):")
        for k in sorted(judge_failed):
            print(f"   {k}: {judge_failed[k]}건")

    # 리포트 생성: 실행 루프 밖에서 samples.jsonl을 1회 로드(여기서만 전체 메모리 사용).
    try:
        from src.report.generate import generate_report

        results_for_report = load_sample_results(run_dir)
        # 이 run에서 **실제로 쓴** 모드·프로파일을 넘긴다(--reasoning-modes/--profile 반영).
        # config를 다시 읽게 두면 리포트가 실행 조건과 다른 값을 싣는다.
        # 모드는 매트릭스에서 유도한다 — 그것이 실제로 돌린 조합의 유일한 진실이다.
        modes_in_run = list(dict.fromkeys(c["reasoning_mode"] for c in matrix)) or None
        report_path = generate_report(
            run_dir, results_for_report, scores, models_cfg,
            reasoning_modes=modes_in_run, profile=profile,
        )
        print(f"리포트 생성: {report_path}")
    except Exception as e:
        print(f"  [리포트 생성 스킵] {type(e).__name__}: {e}")

    try:
        from src.report.index import rebuild_index

        idx = rebuild_index()
        print(f"인덱스 갱신: {idx}")
    except Exception as e:
        print(f"  [인덱스 갱신 스킵] {type(e).__name__}: {e}")

    # ── 종료 코드 판정 ────────────────────────────────────────────────
    # 실패해도 exit 0을 내면 자동화(CI·스크립트·에이전트)가 "성공"으로 오판한다. 실측 위험:
    # 모델 전체가 403/400이어도 "실행 완료" 후 0을 반환해, 빈 리포트가 정상 산출물로 커밋될 수 있다.
    # 리포트는 이미 생성했으므로(부분 결과도 보존 가치가 있다) **결과는 남기고 종료 코드로만 알린다.**
    # 기대 셀 수는 **매트릭스 전체**다. skipped_cells를 빼면 로드 실패로 태스크가 통째로
    # 누락돼도 "기대치를 다 채웠다"가 되어 exit 0이 나온다(그게 옛 동작이었다).
    return _exit_code(
        scores,
        expected_cells=len(matrix),
        failed_tasks=failed_tasks,
        shortfall_tasks=shortfall_tasks,
    )


def _purge_previous_runs(results_dir: Path, reports_dir: Path, *, keep_run_id: str) -> None:
    """--fresh: 이전 run들을 **`.trash/`로 이동**한다(즉시 삭제하지 않는다).

    왜 이동인가: 새 run이 중간에 죽으면 리포트가 하나도 없는 상태가 되고, README의
    '최신 리포트' 링크가 깨진다(실측 사고 2026-08-06 — git에서 복구해야 했다).
    보관해 두면 그 상황에서도 직전 결과를 되찾을 수 있다. `.trash/`는 gitignore 대상이며,
    다음 --fresh 때 이전 보관본을 정리하므로 무한히 쌓이지 않는다.

    `keep_run_id`(이번 run 디렉터리)는 이동 대상에서 제외한다 — 이미 manifest를 써 뒀다.
    """
    import shutil

    trash = Path(".trash") / f"purged-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    moved: list[str] = []

    # 직전 보관본 정리(하나만 유지) — 디스크가 무한히 늘지 않게.
    trash_root = Path(".trash")
    if trash_root.exists():
        for old in sorted(trash_root.glob("purged-*"))[:-1]:
            shutil.rmtree(old, ignore_errors=True)

    for base in (results_dir, reports_dir):
        if not base.exists():
            continue
        for item in sorted(base.iterdir()):
            if item.name == keep_run_id:
                continue
            dest = trash / base.name / item.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(dest))
            moved.append(f"{base.name}/{item.name}")

    if moved:
        print(f"[--fresh] 이전 run {len(moved)}개를 {trash}/로 이동(삭제 아님): "
              f"{', '.join(moved[:6])}{' …' if len(moved) > 6 else ''}")
    else:
        print("[--fresh] 정리할 이전 run이 없습니다.")


def _manifest_drift(prev: dict[str, Any], cur: RunManifest) -> list[str]:
    """resume 시 기존 manifest와 현재 설정의 차이를 사람이 읽을 문장으로 반환(없으면 빈 리스트).

    비교 대상은 **결과 해석을 바꾸는 축**만이다: 모델 집합·태스크 집합·reasoning 모드·
    샘플 수·프로파일. created_at·git_commit은 당연히 다르므로 비교하지 않는다
    (코드 수정 후 실패 셀만 재실행하는 정상 워크플로를 막지 않기 위해).
    """
    out: list[str] = []

    prev_models = {m.get("id") for m in (prev.get("models") or [])}
    cur_models = {m.get("id") for m in cur.models}
    if prev_models and prev_models != cur_models:
        added = sorted(cur_models - prev_models)
        removed = sorted(prev_models - cur_models)
        parts = []
        if added:
            parts.append(f"추가 {added}")
        if removed:
            parts.append(f"제외 {removed}")
        out.append(f"모델 구성이 다릅니다({', '.join(parts)}) — 기존 run: {sorted(prev_models)}")

    prev_tasks = set(prev.get("task_ids") or [])
    cur_tasks = set(cur.task_ids)
    if prev_tasks and prev_tasks != cur_tasks:
        out.append(
            f"태스크 구성이 다릅니다(추가 {sorted(cur_tasks - prev_tasks)}, "
            f"제외 {sorted(prev_tasks - cur_tasks)})"
        )

    prev_modes = list(prev.get("reasoning_modes") or [])
    if prev_modes and prev_modes != list(cur.reasoning_modes):
        out.append(f"reasoning 모드가 다릅니다: {prev_modes} → {list(cur.reasoning_modes)}")

    if prev.get("samples_per_task") and prev["samples_per_task"] != cur.samples_per_task:
        out.append(
            f"태스크당 샘플 수가 다릅니다: {prev['samples_per_task']} → {cur.samples_per_task} "
            f"(셀마다 n이 달라져 비교가 깨집니다)"
        )

    if prev.get("profile") and prev["profile"] != cur.profile:
        out.append(
            f"Databricks 프로파일이 다릅니다: {prev['profile']} → {cur.profile} "
            f"(워크스페이스가 다르면 엔드포인트·단가·지연의 전제가 달라집니다)"
        )
    return out


def _exit_code(
    scores: dict[str, Any],
    expected_cells: int,
    *,
    failed_tasks: dict[str, str] | None = None,
    shortfall_tasks: dict[str, tuple[int, int]] | None = None,
    judge_failure_rate_limit: float = JUDGE_FAILURE_EXIT_RATE,
) -> int:
    """실행 결과로 프로세스 종료 코드를 정한다. 0=정상, 1=자동화가 멈춰야 하는 실패.

    **"완주"의 정의**: 매트릭스의 모든 셀이 요청한 샘플 수만큼 채점됐고, 실패가 임계 미만이며,
    태스크가 하나도 누락되지 않은 상태. 아래 중 하나라도 걸리면 exit 1이다:
    - 태스크 로드·구현 실패로 셀이 아예 실행되지 않음 (`failed_tasks`)
    - 요청보다 적은 샘플로 채점됨 (`shortfall_tasks`) — `--samples 30`은 완주 조건이다
    - 채점 예외(metrics.error) 셀 → 코드·데이터 문제
    - 실패율 과다(unreliable) 셀 → 점수가 장애의 산물
    - 전체 호출 실패율 > GLOBAL_FAILURE_EXIT_RATE → 엔드포인트/권한 문제(403·400 등)
    - 전체 judge 실패율 > judge_failure_rate_limit → 채점기가 제대로 안 돌았다
    - 완료 셀이 기대 매트릭스보다 적음 → 중간에 죽음
    산발적 소수 실패(임계 미만)는 0을 유지한다 — 이 벤치마크에서 흔하고, 그 셀은
    채점에서 제외되며 리포트에 드러나므로 실행 자체를 실패로 볼 필요는 없다.
    """
    reasons: list[str] = []

    if failed_tasks:
        detail = ", ".join(f"{t}({why})" for t, why in sorted(failed_tasks.items()))
        reasons.append(f"실행되지 않은 태스크 {len(failed_tasks)}개: {detail}")

    if shortfall_tasks:
        detail = ", ".join(f"{t} {got}/{want}" for t, (got, want) in sorted(shortfall_tasks.items()))
        reasons.append(
            f"요청보다 적게 채점된 태스크 {len(shortfall_tasks)}개: {detail} "
            f"— 표본이 작아 다른 태스크와 같은 신뢰도로 비교할 수 없다"
        )

    err_cells = [k for k, v in scores.items() if "error" in (v.get("metrics") or {})]
    if err_cells:
        reasons.append(f"채점 오류 셀 {len(err_cells)}개: {', '.join(sorted(err_cells)[:5])}")

    unreliable = [k for k, v in scores.items() if v.get("unreliable")]
    if unreliable:
        reasons.append(f"실패율 과다(신뢰불가) 셀 {len(unreliable)}개: {', '.join(sorted(unreliable)[:5])}")

    total_n = sum(v.get("n", 0) for v in scores.values())
    total_failed = sum(v.get("n_call_failed", 0) for v in scores.values())
    if total_n and total_failed / total_n > GLOBAL_FAILURE_EXIT_RATE:
        reasons.append(
            f"전체 호출 실패율 {total_failed}/{total_n} "
            f"({total_failed / total_n:.1%}) > {GLOBAL_FAILURE_EXIT_RATE:.0%}"
        )

    # judge 실패도 종료 코드에 반영한다 — judge가 대량 실패하면 생성 태스크의 대표 수치가
    # 사실상 비어 있는데도 예전엔 exit 0이었다(실측: max_tokens 부족으로 30/30 실패한 셀).
    judge_ok = judge_failed = 0
    for v in scores.values():
        jd = (v.get("metrics") or {}).get("judge_detail") or {}
        judge_ok += jd.get("n") or 0
        judge_failed += jd.get("n_failed") or 0
    judge_total = judge_ok + judge_failed
    if judge_total and judge_failed / judge_total > judge_failure_rate_limit:
        reasons.append(
            f"judge 실패율 {judge_failed}/{judge_total} "
            f"({judge_failed / judge_total:.1%}) > {judge_failure_rate_limit:.0%} "
            f"— 생성 태스크의 judge 수치를 신뢰할 수 없다"
        )

    if expected_cells and len(scores) < expected_cells:
        reasons.append(f"완료 셀 {len(scores)}/{expected_cells} — 일부 셀이 실행되지 않았다")

    if reasons:
        print("\n" + "=" * 70)
        print("❌ 실행을 실패로 판정합니다(exit 1). 리포트는 생성됐지만 수치를 신뢰할 수 없습니다:")
        for r in reasons:
            print(f"   - {r}")
        print("=" * 70)
        return 1
    return 0


def _run_judge_streaming(inst, samples, run_dir, model_id, mode, judge_client, judge_endpoint):
    """누적기 경로에서 per-task judge를 수행.

    judge_scores()는 parsed 리스트를 요구하지만, 스트리밍에선 parsed를 보관하지 않으므로
    이 시점에 samples.jsonl에서 해당 셀의 model_output을 읽어 parsed를 재구성한다(디스크 기반, O(샘플)이지만
    한 셀 범위라 작음). 태스크의 judge_scores 계약을 그대로 사용한다.

    **호출이 실패한 샘플은 judge에 보내지 않는다** (2026-08-06 수정). 예전엔 저장된
    `"__ERROR__: FMAPIError ..."` 문자열을 다시 parse_output에 넘겨, 그 오류 메시지를
    모델 답변으로 취급해 judge에게 채점을 요청했다(judge 비용 낭비 + 오류 문구에 대한
    무의미한 점수). 실패 샘플은 아예 제외하고, 그 사실을 judge 결과에 남긴다.
    """
    from src.results import load_sample_results

    rows = [
        r for r in load_sample_results(run_dir)
        if r.model_id == model_id and r.task_id == inst.task_id and r.reasoning_mode == mode
    ]
    by_sid = {r.sample_id: r for r in rows}
    parsed: list[Any] = []
    kept_samples = []
    n_call_failed = 0
    for s in samples:
        r = by_sid.get(s.sample_id)
        # finish_reason이 error이거나 행 자체가 없으면 호출 실패 → judge 대상 아님.
        if r is None or r.finish_reason == "error":
            n_call_failed += 1
            continue
        try:
            parsed.append(inst.parse_output(r.model_output, s))
        except Exception:
            parsed.append(None)   # 파싱 실패는 judge가 낮게 평가하도록 그대로 넘긴다
        kept_samples.append(s)

    if not kept_samples:
        # 셀 전체가 호출 실패 — judge를 호출할 이유가 없다(비용·시간 낭비).
        return {"judge_mean": None, "judge_scores": [], "n_judged": 0,
                "n_judge_skipped_call_failed": n_call_failed}

    result = inst.judge_scores(parsed, kept_samples, judge_client, judge_endpoint)
    if n_call_failed and isinstance(result, dict):
        result["n_judge_skipped_call_failed"] = n_call_failed
    return result


def _atomic_write_json(path: Path, obj: Any) -> None:
    """tmp에 쓰고 os.replace로 교체(부분 기록 방지). scores.json 체크포인트용."""
    import os

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _poisoned_cells(samples_path: Path, threshold: float = UNRELIABLE_FAILURE_RATE) -> set[str]:
    """samples.jsonl을 훑어 모델 응답이 __ERROR__인 비율이 threshold를 넘는 셀 key를 반환.

    IP ACL 403·엔드포인트 장애 등으로 한 셀의 응답이 대량 실패하면, 그 셀의 지표는
    0.0류로 산출되지만 예외가 아니라 scores.json에 'error' 키 없이 남는다. resume가 그런
    셀을 '완료'로 오판해 스킵하는 것을 막기 위해, 여기서 오염 셀을 식별해 재실행 대상으로 삼는다.
    파일이 없으면 빈 집합.
    """
    if not samples_path.exists():
        return set()
    total: dict[str, int] = {}
    errs: dict[str, int] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                key = f"{d['model_id']}::{d['task_id']}::{d['reasoning_mode']}"
            except (json.JSONDecodeError, KeyError):
                continue
            total[key] = total.get(key, 0) + 1
            if str(d.get("model_output", "")).startswith("__ERROR__"):
                errs[key] = errs.get(key, 0) + 1
    return {k for k, n in total.items() if n > 0 and errs.get(k, 0) / n > threshold}


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
                    # revision(HF 커밋 SHA)까지 스냅샷해야 이 run이 어떤 데이터로 돌았는지
                    # 사후에 확정할 수 있다. 로더가 실제로 이 값으로 고정해 로드한다.
                    "revision": entry.get("revision"),
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

    **judge_mean이 None/비수치면 키를 아예 넣지 않는다** — 0.0으로 채우면 정량표에서
    "judge가 최악으로 평가함"처럼 읽혀 성능 저하로 오독된다(옛 IMG-1 judge_mean=0.0 사례).
    대신 실패 건수(n_judge_failed)를 detail에 남겨 왜 값이 없는지 추적 가능하게 한다.
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
    failed = jr.get("n_judge_failed")
    if failed:
        detail["n_failed"] = failed
    # 호출 실패로 judge에 보내지도 않은 샘플 수 — judge 실패(n_failed)와 구분해 남긴다.
    # 둘을 합치면 "채점기 문제"와 "엔드포인트 문제"가 섞여 원인 파악이 어렵다.
    skipped = jr.get("n_judge_skipped_call_failed")
    if skipped:
        detail["n_skipped_call_failed"] = skipped
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


def _elide_middle(text: str, limit: int) -> str:
    """가운데를 잘라 head+tail을 모두 보존(프롬프트 저장용).

    프롬프트는 '지시…큰 컨텍스트(표/문서)…질문' 구조라 앞만 자르면(head-only) 뒤쪽 질문이
    사라진다(리포트 갤러리에서 '질문이 안 보임' 원인). head 2/3 + tail 1/3을 남겨 질문 보존.
    """
    if len(text) <= limit:
        return text
    head = int(limit * 2 / 3)
    tail = limit - head
    return text[:head] + "\n…(중략)…\n" + text[-tail:]


if __name__ == "__main__":
    sys.exit(main())
