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
from collections.abc import Callable
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
# 중단 시 이미 과금된 호출 결과가 통째로 사라진다.
# **이 값이 실제 복구 단위가 되려면 resume이 부분 행을 재사용해야 한다** — `_partial_progress`가
# 그 역할을 한다(2026-08-06). 둘이 함께 있을 때만 "중단 시 재호출 손실 ≤ 4개"가 성립한다.
# flush만 있고 부분 재사용이 없던 동안에는 resume이 미완료 셀 행을 전부 버려 30개를
# 다시 호출했다(그때의 "손실 최대 4개"라는 서술은 사실이 아니었다).
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
        # make_run_id가 디렉터리 생성으로 **원자적 예약**까지 한다(동시 실행 충돌 방지).
        # 따라서 여기서 mkdir을 다시 부르지 않는다 — 부르면 예약 의미가 사라진다.
        run_id = make_run_id(results_root=args.out)
        run_dir = Path(args.out) / run_id

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
    if resuming:
        # **manifest는 resume의 전제조건이다.** 예전엔 `exists()`일 때만 검증해서, 파일이
        # 없으면 그냥 새 manifest를 써 버렸다 — 즉 "구성을 확인할 수 없는" 최악의 경우가
        # 가장 느슨하게 통과했다. 없으면 거부한다.
        if not manifest_path.exists():
            print(f"\n❌ --resume 거부: {manifest_path}가 없습니다.\n"
                  f"   어떤 구성으로 돌던 run인지 확인할 수 없어, 지금 설정이 다르면 서로 다른\n"
                  f"   구성의 결과가 한 run에 섞입니다.\n"
                  f"   → 새 run으로 전체 재실행하세요(--resume 없이).", file=sys.stderr)
            return 1
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(prev, dict):
                raise ValueError(f"manifest 최상위가 dict가 아님({type(prev).__name__})")
            missing_axes = _missing_manifest_axes(prev)
            if missing_axes:
                raise ValueError(
                    f"구성 축이 비어 있음: {missing_axes} — 이 값이 없으면 구성이 같은지 "
                    f"비교할 수 없습니다"
                )
        except Exception as e:
            # **fail-closed.** 읽을 수 없거나 핵심 축이 없으면 거부한다.
            print(f"\n❌ --resume 거부: manifest.json을 신뢰할 수 없습니다 "
                  f"({type(e).__name__}: {e}).\n"
                  f"   구성이 같은지 확인할 수 없으면 옛 결과와 섞여 리포트가 거짓이 됩니다.\n"
                  f"   → 새 run으로 전체 재실행하세요(--resume 없이).", file=sys.stderr)
            return 1
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

    # ⚠️ --fresh 정리는 여기서도 하지 않는다. **새 리포트가 생성된 뒤**(_run_samples 끝)에
    # 해야 중단 시 README '최신 리포트' 링크와 index가 깨지지 않는다 — 실행 전에 치우면
    # 첫 셀에서 죽어도 이전 리포트가 이미 사라진다(2026-08-06 재지적).
    # _run_samples가 리포트·인덱스를 만든 직후 정리하고, 인덱스를 한 번 더 재생성한다.

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
            fresh=args.fresh,
            results_root=Path(args.out),
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
    fresh: bool = False,
    results_root: Path | None = None,
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
        # samples.jsonl 정합화 + **부분 셀 이어달리기 준비**.
        # 완료 셀(scores에 존재)의 행은 그대로 두고, 미완료 셀의 행은 **보존한 뒤**
        # 이미 호출한 sample_id를 뽑아 둔다(아래 partial_done). 그러면 중단된 셀을
        # 처음부터 다시 호출하지 않는다 — 예전엔 미완료 행을 전부 버려서 5샘플마다
        # flush해도 결국 30개를 다시 호출했다(2026-08-06 지적).
        # 단 오염 셀(>50% __ERROR__)의 행은 버린다 — 그 응답들은 재시도 대상이다.
        partial_done = _partial_progress(samples_path, set(scores.keys()), drop_keys=poisoned)
        if partial_done:
            total_reusable = sum(len(v) for v in partial_done.values())
            print(f"  [resume] 부분 완료 셀 {len(partial_done)}개에서 {total_reusable}샘플 재사용"
                  f"(재호출 생략): "
                  + ", ".join(f"{k}({len(v)})" for k, v in sorted(partial_done.items())[:4]))
    else:
        scores = {}
        partial_done = {}
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

    # judge 클라이언트(공유, 넉넉한 timeout). 실패해도 정량은 진행하되 종료 코드에 반영한다.
    judge_client = None
    judge_init_error: str | None = None
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
            # judge 클라이언트 초기화 실패를 조용히 넘기면 judge 태스크가 전부 채점 없이
            # "완주"한다(실측: rc=0). 이유를 기록해 종료 코드에 반영한다 — judge를 의도적으로
            # 끄려면 `--no-judge`를 쓰라는 신호이기도 하다.
            judge_init_error = f"{type(e).__name__}: {e}"
            print(f"  [judge 비활성 — 클라이언트 초기화 실패] {judge_init_error}")
            print("    ⚠️  judge 대상 태스크가 채점 없이 진행됩니다 → 실행을 실패로 판정합니다"
                  "(의도적으로 끄려면 --no-judge를 쓰세요).")

    executed = skipped_cells = resume_skipped = 0
    # 요청보다 적게 로드된 태스크: {task_id: (로드됨, 요청됨)}. 종료 요약·셀 메타에 쓴다.
    shortfall_tasks: dict[str, tuple[int, int]] = {}
    # 아예 실행하지 못한 태스크: {task_id: 이유}. **종료 코드를 실패로 만든다** —
    # 태스크가 통째로 누락됐는데 "실행 완료 exit 0"이 되던 문제를 막는다.
    failed_tasks: dict[str, str] = {}
    # judge 채점을 받아야 하는 셀 key 집합. 실제로 judge_mean이 붙었는지 종료 시 대조한다 —
    # judge 클라이언트 초기화 실패·judge_scores 예외로 채점이 빠져도 exit 0이던 문제 방지.
    judge_expected_cells: set[str] = set()
    # 필수 산출물(report.md·index·presentation·chart) 생성 실패 목록 → 종료 코드에 반영(P1-5).
    artifact_errors: list[str] = []
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
                # 이 셀이 judge 채점을 **받아야 하는지**(want_judge)와 실제로 돌릴 수
                # 있는지(do_judge)를 구분한다. judge가 필요한데 못 돌린 셀은 완주가 아니다.
                expects_judge = enable_judge and want_judge and hasattr(inst, "judge_scores")
                do_judge = expects_judge and judge_client is not None
                if expects_judge:
                    judge_expected_cells.add(key)
                cell_results: list[SampleResult] = []
                n_call_failed = 0   # 이 셀의 호출 실패 수 → 점수와 함께 기록(아래 주석 참고)
                # 누적기 add 실패 수. 경고만 찍고 넘어가면 그 샘플이 분모에서 조용히 빠져
                # (n_evaluated가 줄어든) 셀이 "완주"로 통과한다 → 셀 오류로 승격시킨다.
                n_acc_errors = 0

                # 부분 이어달리기: 이 셀에서 **이미 호출해 저장된** sample_id는 다시 부르지 않고
                # samples.jsonl에서 결과를 읽어 채점에만 넣는다(중단 시 재호출 비용 절감).
                reuse_sids = partial_done.get(key, set())
                reused_rows: dict[int, Any] = {}
                if reuse_sids:
                    # samples.jsonl을 **스트리밍으로 한 줄씩** 훑어 이 셀의 행만 뽑는다.
                    # 예전엔 `load_sample_results(run_dir)`(전체 파일을 리스트로 로드)를 셀
                    # 루프 안에서 호출해 두 가지가 깨졌다: (1) 파일이 커지는 동안 셀마다 전체
                    # 재스캔 → O(셀수 × 행수), (2) 전 행을 메모리에 올려 러너의 O(1) 메모리
                    # 계약 위반. 여기서는 해당 셀 행만 dict에 담는다(최대 샘플 수개).
                    reused_rows = _load_cell_rows(
                        samples_path, model.id, task_id, mode, reuse_sids
                    )
                    print(f"  [{model.id}/{task_id}/{mode}] {len(samples)}샘플 "
                          f"(이어달리기: {len(reused_rows)}개 재사용, "
                          f"{len(samples) - len(reused_rows)}개 호출)...", flush=True)
                else:
                    print(f"  [{model.id}/{task_id}/{mode}] {len(samples)}샘플 실행...", flush=True)

                for s in samples:
                    # 이미 호출한 샘플: 저장된 결과로 채점만 하고 API 호출을 건너뛴다.
                    prev = reused_rows.get(s.sample_id)
                    if prev is not None:
                        try:
                            parsed_prev = inst.parse_output(prev.model_output, s)
                        except Exception:
                            parsed_prev = None
                        if acc is not None:
                            try:
                                acc.add(parsed_prev, s)
                            except Exception as e:
                                n_acc_errors += 1
                                print(f"    [acc.add 실패(재사용)] {key} s{s.sample_id}: "
                                      f"{type(e).__name__}: {e}")
                        else:
                            buf_parsed.append(parsed_prev)
                            buf_samples.append(s)
                        continue

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
                            # add 실패를 조용히 삼키면 그 샘플이 분모에서 빠져 n_evaluated가
                            # 어긋난 채 셀이 완주로 통과한다 → 세어서 아래에서 오류로 승격.
                            n_acc_errors += 1
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

                # 누적기 add가 하나라도 실패했으면 이 셀의 분모가 조용히 줄어든 상태다.
                # 수치는 그대로 계산되지만 "몇 개로 낸 값인지"가 어긋나므로, 오류로 표시해
                # 완주 판정·리포트 순위에서 빠지게 한다(경고만 찍던 옛 동작의 구멍).
                if n_acc_errors and isinstance(metrics, dict):
                    metrics["error"] = (
                        f"누적기 add 실패 {n_acc_errors}/{len(samples)}건 — 분모가 어긋나 "
                        f"수치를 신뢰할 수 없음"
                        + (f" (기존 오류: {metrics['error']})" if "error" in metrics else "")
                    )

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
        # 리포트는 이 벤치마크의 **산출물 그 자체**다. 실패를 로그만 남기고 exit 0을 내면
        # "성공했는데 리포트가 없는" 상태가 되어 자동화가 그대로 커밋·푸시한다(P1-5).
        artifact_errors.append(f"report.md: {type(e).__name__}: {e}")
        print(f"  [리포트 생성 실패] {type(e).__name__}: {e}")

    try:
        from src.report.index import rebuild_index

        idx = rebuild_index()
        print(f"인덱스 갱신: {idx}")
    except Exception as e:
        artifact_errors.append(f"reports/index.md: {type(e).__name__}: {e}")
        print(f"  [인덱스 갱신 실패] {type(e).__name__}: {e}")

    # 필수 산출물이 실제로 파일로 존재하는지 확인한다(예외 없이 조용히 안 만들어진 경우 대비).
    reports_dir = Path("reports") / run_dir.name
    for required in ("report.md", "facts.json", "presentation.html"):
        if not (reports_dir / required).exists():
            artifact_errors.append(f"{required} 없음")
    if not list(reports_dir.glob("chart_*.png")):
        artifact_errors.append("chart_*.png 없음")

    # ── 종료 코드 판정 ────────────────────────────────────────────────
    # 실패해도 exit 0을 내면 자동화(CI·스크립트·에이전트)가 "성공"으로 오판한다. 실측 위험:
    # 모델 전체가 403/400이어도 "실행 완료" 후 0을 반환해, 빈 리포트가 정상 산출물로 커밋될 수 있다.
    # 리포트는 이미 생성했으므로(부분 결과도 보존 가치가 있다) **결과는 남기고 종료 코드로만 알린다.**
    # 기대 셀 수는 **매트릭스 전체**다. skipped_cells를 빼면 로드 실패로 태스크가 통째로
    # 누락돼도 "기대치를 다 채웠다"가 되어 exit 0이 나온다(그게 옛 동작이었다).
    #
    # **정리보다 먼저 판정한다** — 아래 --fresh 보관 이동이 이 판정 결과를 봐야 하기 때문이다.
    code = _exit_code(
        scores,
        expected_cells=len(matrix),
        expected_keys={_cell_key(c["model_id"], c["task_id"], c["reasoning_mode"])
                       for c in matrix},
        failed_tasks=failed_tasks,
        shortfall_tasks=shortfall_tasks,
        judge_expected_cells=judge_expected_cells,
        judge_init_error=judge_init_error,
        artifact_errors=artifact_errors,
    )

    # ── --fresh 정리는 **맨 마지막에, 판정이 통과한 경우에만** ─────────────
    # 이전엔 필수 산출물 존재만 확인하고 치웠다. 그러면 모델 전체가 403이어서 점수가 텅 빈
    # run도 구조적으로는 report.md·chart를 만들어내므로, **검증된 이전 run을 전부 보관 이동한 뒤
    # exit 1**을 내는 최악의 조합이 나온다(믿을 수 있는 유일한 리포트를 스스로 치운다).
    # 지금은 exit 0(= 수치를 신뢰할 수 있음)일 때만 치운다. 실행 중간에 죽으면 정리가 아예
    # 일어나지 않아 README '최신 리포트' 링크·index가 그대로 살아 있다.
    if fresh:
        if code != 0:
            print("\n[--fresh 보류] 이번 run이 실패로 판정돼 이전 run을 그대로 둡니다 "
                  "(신뢰할 수 있는 리포트를 잃지 않기 위해). 원인을 고친 뒤 다시 실행하세요.")
        else:
            # 정리 **후** 단계(인덱스 재생성)까지 한 트랜잭션으로 묶는다. 그 단계가 실패하면
            # 이동을 되돌린다 — 안 그러면 "검증된 이전 리포트를 옮긴 뒤 exit 1"이 그대로
            # 남는다(정리 전 판정만으로는 이 조합을 막을 수 없다).
            def _post_move_verify() -> None:
                from src.report.index import rebuild_index

                print(f"인덱스 재생성(정리 후): {rebuild_index()}")

            try:
                _purge_previous_runs(
                    results_root or Path("results"), Path("reports"),
                    keep_run_id=run_dir.name, post_move_verify=_post_move_verify,
                )
            except Exception as e:
                # 롤백이 끝난 상태다(이전 run은 제자리). 이번 run의 결과·리포트는 그대로
                # 남아 있지만 정리가 안 됐으므로 자동화가 "완전히 깨끗하다"고 보면 안 된다.
                print(f"\n[--fresh 실패] 정리를 되돌렸습니다({type(e).__name__}: {e}). "
                      f"이전 run은 제자리에 있고 이번 run의 리포트도 남아 있습니다.")
                return 1

    return code


def _purge_previous_runs(
    results_dir: Path,
    reports_dir: Path,
    *,
    keep_run_id: str,
    post_move_verify: Callable[[], None] | None = None,
) -> None:
    """--fresh: 이전 run들을 **`.trash/`로 이동**한다(즉시 삭제하지 않는다).

    왜 이동인가: 새 run이 중간에 죽으면 리포트가 하나도 없는 상태가 되고, README의
    '최신 리포트' 링크가 깨진다(실측 사고 2026-08-06 — git에서 복구해야 했다).
    보관해 두면 그 상황에서도 직전 결과를 되찾을 수 있다. `.trash/`는 gitignore 대상이며,
    다음 --fresh 때 이전 보관본을 정리하므로 무한히 쌓이지 않는다.

    `keep_run_id`(이번 run 디렉터리)는 이동 대상에서 제외한다 — 이미 manifest를 써 뒀다.

    **전부 성공하거나 전부 되돌린다(트랜잭션).** 중간에 이동이 실패하거나
    `post_move_verify`(정리 후 index 재생성 등)가 실패하면, 이미 옮긴 것을 제자리로
    되돌리고 예외를 올린다. 그러지 않으면 "검증된 이전 리포트를 옮긴 뒤 실패"라는
    최악의 상태가 남는다 — 정리 전에 판정을 해도 정리 *후* 단계는 막을 수 없다.
    """
    import shutil

    trash_root = Path(".trash")
    # 이번 보관본 디렉터리는 **원자적으로 예약**한다 — 같은 초에 두 번 정리하면 같은 이름을
    # 공유해 한쪽이 다른쪽 트리 안에 중첩되거나 중간에 실패한다.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trash = trash_root / f"purged-{stamp}"
    for n in range(1, 100):
        cand = trash if n == 1 else trash_root / f"purged-{stamp}-{n}"
        try:
            cand.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        trash = cand
        break

    moved: list[str] = []
    done: list[tuple[Path, Path]] = []      # (원래 위치, 보관 위치) — 롤백용
    # 콜백 직전의 디렉터리 스냅샷. 콜백(index 재생성 등)이 **새로 만든 경로**를 롤백 때
    # 걷어내기 위한 기준이다. 이게 없으면 콜백이 index.md를 다시 쓴 뒤 실패했을 때,
    # 복원 조건(`not src.exists()`)이 거짓이 되어 원본 index를 조용히 못 되살린다
    # (실측: old report는 복원됐는데 live index.md는 반쪽, 원본은 .trash에 남았다).
    snapshot: dict[Path, set[str]] = {}
    try:
        for base in (results_dir, reports_dir):
            if not base.exists():
                continue
            for item in sorted(base.iterdir()):
                if item.name == keep_run_id:
                    continue
                dest = trash / base.name / item.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(dest))
                done.append((item, dest))
                moved.append(f"{base.name}/{item.name}")
        if post_move_verify is not None:
            snapshot = {
                base: {p.name for p in base.iterdir()}
                for base in (results_dir, reports_dir) if base.exists()
            }
            post_move_verify()
    except Exception:
        # 되돌린다. 롤백이 완전하지 않으면 그 사실을 크게 알린다 — 보관본 경로만 알면
        # 사람이 복구할 수 있으므로, 조용히 삼켜서 "사라진 것처럼" 보이게 하지 않는다.
        # ① 콜백이 만든 잔여물을 먼저 걷어낸다(원본을 덮어쓸 자리를 비운다).
        for base, before in snapshot.items():
            if not base.exists():
                continue
            for p in base.iterdir():
                if p.name in before:
                    continue                  # 콜백 전에도 있던 것 — 건드리지 않는다
                try:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                except OSError as ce:
                    print(f"  ❗ 콜백 잔여물 제거 실패: {p} ({type(ce).__name__}: {ce})")

        # ② 옮긴 것을 제자리로. 자리가 차 있으면(콜백이 같은 경로를 다시 썼으면) 그 잔여물을
        #    치우고 원본을 복원한다 — 옮겨둔 원본이 더 신뢰할 수 있는 상태다.
        restored = 0
        for src, dest in reversed(done):
            try:
                if not dest.exists():
                    continue
                if src.exists():
                    shutil.rmtree(src) if src.is_dir() else src.unlink()
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(src))
                restored += 1
            except OSError as re:
                print(f"  ❗ 롤백 실패: {dest} → {src} ({type(re).__name__}: {re})")

        if restored != len(done):
            print(f"  ❗❗ [--fresh 롤백 불완전] {len(done)}개 중 {restored}개만 복원했습니다. "
                  f"복원되지 않은 것은 **{trash}/** 에 그대로 있으니 수동으로 되돌리세요.")
        elif done:
            print(f"  [--fresh 롤백] 이동한 {len(done)}개를 모두 제자리로 되돌렸습니다.")
        raise

    if moved:
        print(f"[--fresh] 이전 run {len(moved)}개를 {trash}/로 이동(삭제 아님): "
              f"{', '.join(moved[:6])}{' …' if len(moved) > 6 else ''}")
    else:
        print("[--fresh] 정리할 이전 run이 없습니다.")

    # 옛 보관본 정리는 **이번 보관본이 완성된 뒤에** 한다(하나만 유지 — 디스크 무한 증가 방지).
    # 순서를 뒤집으면(옛것 삭제 → 이동) 이동이 중간에 실패했을 때 옛 보관본도 없고 이번
    # 보관본도 반쪽인 상태가 된다. 삭제 실패는 조용히 넘기지 않고 알린다(누수 신호).
    if trash_root.exists():
        for old in sorted(trash_root.glob("purged-*")):
            if old == trash:
                continue
            try:
                shutil.rmtree(old)
            except OSError as e:
                print(f"  [.trash 정리 실패] {old}: {type(e).__name__}: {e} — 수동 삭제 필요")


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


# resume 전에 manifest에 반드시 있어야 하는 구성 축. 이 값이 없으면 "구성이 같은가"를
# 물을 수 없고, `_manifest_drift`는 없는 필드를 건너뛰므로 조용히 통과한다.
RESUME_REQUIRED_AXES = ("models", "task_ids", "reasoning_modes", "samples_per_task", "profile")


def _missing_manifest_axes(prev: dict[str, Any]) -> list[str]:
    """resume 대상 manifest에서 **쓸 수 없는** 핵심 축 목록(문제 없으면 빈 리스트).

    유효 JSON인 `{}`도 파싱은 되지만 드리프트 검사를 전부 통과해버린다 —
    모델·태스크·샘플 수가 달라도 옛 scores와 섞이게 된다.

    단순 emptiness 비교로는 부족하다: 다섯 축을 모두 `0`으로 둔 manifest는
    `0 in (None, [], {}, "")`이 거짓이라 통과하고, 드리프트 검사도 타입이 안 맞아 비교를
    건너뛰므로 **구성이 다른 run도 resume이 된다**(재검토 실측). 그래서 축마다 타입·값까지 본다.
    """
    bad: list[str] = []

    def _nonempty_str_list(v: Any) -> bool:
        return isinstance(v, list) and bool(v) and all(
            isinstance(x, str) and x.strip() for x in v
        )

    # models: [{"id": "..."}] 형태여야 한다 — 드리프트가 id 집합을 비교하기 때문이다.
    models = prev.get("models")
    if not (isinstance(models, list) and models and all(
            isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip()
            for m in models)):
        bad.append("models")

    for axis in ("task_ids", "reasoning_modes"):
        if not _nonempty_str_list(prev.get(axis)):
            bad.append(axis)

    # samples_per_task: 양의 정수. bool은 int의 서브클래스라 True가 1로 통과하므로 배제한다.
    spt = prev.get("samples_per_task")
    if isinstance(spt, bool) or not isinstance(spt, int) or spt <= 0:
        bad.append("samples_per_task")

    profile = prev.get("profile")
    if not (isinstance(profile, str) and profile.strip()):
        bad.append("profile")

    return bad


def _cell_key(model_id: str, task_id: str, mode: str) -> str:
    """셀(모델×태스크×모드)의 고유 키. scores.json의 키이자 samples 행의 그룹 키다."""
    return f"{model_id}::{task_id}::{mode}"


def _exit_code(
    scores: dict[str, Any],
    expected_cells: int,
    *,
    expected_keys: set[str] | None = None,
    failed_tasks: dict[str, str] | None = None,
    shortfall_tasks: dict[str, tuple[int, int]] | None = None,
    judge_expected_cells: set[str] | None = None,
    judge_init_error: str | None = None,
    artifact_errors: list[str] | None = None,
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

    if artifact_errors:
        # 리포트·인덱스·프레젠테이션은 이 벤치마크의 산출물이다. 없으면 실행은 실패다.
        reasons.append(
            f"필수 산출물 생성 실패 {len(artifact_errors)}건: {', '.join(artifact_errors[:3])}"
        )

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

    # judge 자체가 **돌지 않은** 경우를 먼저 잡는다. 클라이언트 초기화 실패나 judge_scores
    # 예외로 채점이 빠져도 예전엔 exit 0이었다(실측: judge_error만 있는 셀 → rc=0).
    # `--no-judge`를 쓰지 않았다면 judge 대상 셀에는 judge_mean/judge_detail이 있어야 한다.
    if judge_init_error:
        reasons.append(
            f"judge 클라이언트 초기화 실패로 judge 채점이 전부 빠졌습니다({judge_init_error}) "
            f"— 의도적으로 끄려면 --no-judge를 사용하세요"
        )
    if judge_expected_cells:
        missing_judge: list[str] = []
        judge_errors: list[str] = []
        for key in sorted(judge_expected_cells):
            entry = scores.get(key)
            if not entry or not isinstance(entry.get("metrics"), dict):
                continue   # 셀 자체가 없으면 '완료 셀 부족' 조건이 잡는다
            m = entry["metrics"]
            if m.get("judge_error"):
                judge_errors.append(f"{key}({str(m['judge_error'])[:60]})")
            elif "judge_mean" not in m and not (m.get("judge_detail") or {}).get("n"):
                missing_judge.append(key)
        if judge_errors:
            reasons.append(
                f"judge 채점 예외 셀 {len(judge_errors)}개: {', '.join(judge_errors[:3])}"
            )
        if missing_judge:
            reasons.append(
                f"judge 대상인데 채점 결과가 없는 셀 {len(missing_judge)}개: "
                f"{', '.join(missing_judge[:5])}"
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

    # 개수 비교가 아니라 **키 집합**을 비교한다. 개수만 보면 옛 run에서 남은 scores 항목이
    # 이번 매트릭스의 누락 셀을 벌충해 "다 돌았다"가 되고(exit 0), 그 낡은 항목이 리포트
    # 순위에도 들어간다(--resume + 구성 변경 시 실제로 가능한 경로).
    if expected_keys:
        missing = sorted(expected_keys - set(scores))
        extra = sorted(set(scores) - expected_keys)
        if missing:
            reasons.append(
                f"누락 셀 {len(missing)}/{len(expected_keys)} — 실행되지 않았다: "
                f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}"
            )
        if extra:
            reasons.append(
                f"이번 매트릭스에 없는 scores 항목 {len(extra)}개 — 옛 run의 결과가 섞였다: "
                f"{', '.join(extra[:5])}{' …' if len(extra) > 5 else ''}"
            )
    elif expected_cells and len(scores) < expected_cells:
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
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        # 실패 시 tmp를 남기지 않는다 — 셀마다 호출되므로 남으면 계속 쌓이고,
        # 수동 점검 때 어느 게 진짜 결과인지 헷갈린다. 원본은 그대로 유지된다.
        tmp.unlink(missing_ok=True)
        raise


def _poisoned_cells(samples_path: Path, threshold: float = UNRELIABLE_FAILURE_RATE) -> set[str]:
    """samples.jsonl을 훑어 모델 응답이 __ERROR__인 비율이 threshold를 넘는 셀 key를 반환.

    IP ACL 403·엔드포인트 장애 등으로 한 셀의 응답이 대량 실패하면, 그 셀의 지표는
    0.0류로 산출되지만 예외가 아니라 scores.json에 'error' 키 없이 남는다. resume가 그런
    셀을 '완료'로 오판해 스킵하는 것을 막기 위해, 여기서 오염 셀을 식별해 재실행 대상으로 삼는다.
    파일이 없으면 빈 집합.

    **물리 행이 아니라 샘플 단위(last-wins)로 센다.** append-only라 재실행이 겹치면 같은
    샘플의 옛 성공 행과 새 실패 행이 함께 남는데, 물리 행을 세면 옛 성공이 새 실패를
    희석한다(실측 시나리오: 옛 성공 30 + 최신 403 30 → 실제는 30/30 실패인데 30/60=50%로
    계산돼 임계를 넘지 못하고 오염 셀에서 빠졌다). 채점·리포트가 모두 마지막 행을 쓰므로
    오염 판정도 같은 기준이어야 한다.
    """
    if not samples_path.exists():
        return set()
    # (cell_key, sample_id) → 마지막 행이 실패였는지
    last_failed: dict[tuple[str, int], bool] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if not isinstance(d, dict):
                    continue
                key = f"{d['model_id']}::{d['task_id']}::{d['reasoning_mode']}"
                sid = d["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            last_failed[(key, sid)] = str(d.get("model_output", "")).startswith("__ERROR__")

    total: dict[str, int] = {}
    errs: dict[str, int] = {}
    for (key, _sid), failed in last_failed.items():
        total[key] = total.get(key, 0) + 1
        if failed:
            errs[key] = errs.get(key, 0) + 1
    return {k for k, n in total.items() if n > 0 and errs.get(k, 0) / n > threshold}


def _load_cell_rows(
    samples_path: Path,
    model_id: str,
    task_id: str,
    mode: str,
    sample_ids: set[int],
) -> dict[int, Any]:
    """samples.jsonl을 스트리밍으로 훑어 **한 셀의 지정 샘플 행만** 돌려준다.

    반환: {sample_id: SampleResult 유사 객체(필요 필드만 있는 경량 레코드)}.

    왜 전체 로드를 쓰지 않는가: `load_sample_results()`는 파일 전체를 리스트로 만든다.
    그걸 셀 루프 안에서 부르면 (1) 파일이 커지는 동안 셀마다 전체 재스캔이 되어
    O(셀수 × 행수)가 되고, (2) 모든 행이 메모리에 올라와 러너의 O(1) 메모리 설계가 깨진다.
    여기서는 한 줄씩 읽고 이 셀에 해당하는 행만 남긴다(최대 샘플 수개).
    """
    from src.results import SampleResult

    out: dict[int, Any] = {}
    if not samples_path.exists() or not sample_ids:
        return out
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # `[]`·`null`처럼 **유효 JSON이지만 object가 아닌** 행도 있다. 가드가 없으면
            # 아래 .get에서 AttributeError가 나 resume 전체가 죽는다(로드 경로마다 손상 행
            # 처리가 갈리던 원인). load_sample_results와 같은 기준으로 건너뛴다.
            if not isinstance(d, dict):
                continue
            if (d.get("model_id") != model_id or d.get("task_id") != task_id
                    or d.get("reasoning_mode") != mode):
                continue
            sid = d.get("sample_id")
            if sid not in sample_ids:
                continue
            # 알려진 필드만 넘긴다 — 스키마가 늘어난 옛 파일도 안전하게 읽힌다.
            known = {f: d.get(f) for f in SampleResult.__dataclass_fields__ if f in d}
            try:
                out[sid] = SampleResult(**known)
            except TypeError:
                continue   # 필수 필드가 빠진 손상 행은 건너뛴다(재호출됨)
    return out


def _partial_progress(
    samples_path: Path, done_keys: set[str], *, drop_keys: set[str] | None = None
) -> dict[str, set[int]]:
    """미완료 셀의 **이미 호출한 sample_id**를 셀별로 모아 돌려준다(부분 이어달리기용).

    resume의 핵심 문제였다: 예전 `_reconcile_samples_with_scores`는 scores에 없는 셀의 행을
    전부 버려서, 5샘플마다 flush해도 중단된 셀은 처음부터 30개를 다시 호출했다(비용·시간 낭비).
    이제 그 행들을 **보존**하고, 어디까지 했는지를 이 함수가 알려준다.

    제외 규칙:
    - `done_keys`(완료 셀)는 대상 아님 — 어차피 resume-skip된다.
    - `drop_keys`(오염 셀, >50% __ERROR__)는 행을 재사용하지 않는다 — 그 응답들은 재시도 대상.
    - 실패 응답(`finish_reason == "error"`)은 재사용하지 않는다 — 재시도해야 정상값을 얻는다.

    부작용으로 samples.jsonl을 재작성한다: 완료 셀 + 재사용 가능한 부분 행만 남긴다
    (실패 행·오염 셀 행을 지워, 재실행 결과와 중복되지 않게).
    """
    import os

    if not samples_path.exists():
        return {}
    drop = drop_keys or set()

    kept: list[str] = []
    progress: dict[str, set[int]] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if not isinstance(d, dict):
                    continue   # `[]`·`null` 등 유효 JSON 비-object 행(아래 첨자에서 TypeError)
                key = f"{d['model_id']}::{d['task_id']}::{d['reasoning_mode']}"
                sid = d["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if key in done_keys:
                kept.append(line)          # 완료 셀 행은 그대로 보존
                continue
            if key in drop or d.get("finish_reason") == "error":
                continue                   # 재시도 대상 → 행을 버린다
            kept.append(line)              # 부분 완료 행 보존
            progress.setdefault(key, set()).add(sid)

    # 재작성 실패 시 **원본을 그대로 두고** 이어달리기를 포기한다(재호출은 손해지만,
    # 반쪽 파일로 채점이 오염되는 것보다 안전하다). tmp도 남기지 않는다.
    tmp = samples_path.with_suffix(samples_path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, samples_path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        # 전부 재호출하면 같은 (모델·태스크·모드·샘플) 행이 파일에 두 번 남는다. 그래도
        # 집계가 중복되지 않는 이유: `load_sample_results()`가 그 키로 **마지막 행만** 남기고
        # 읽는다(= 실제로 채점에 쓴 새 응답). 반쪽 파일로 채점하는 것보다 이 편이 안전하다.
        print(f"  [resume] samples.jsonl 재작성 실패({type(e).__name__}: {e}) — "
              f"이어달리기를 건너뛰고 해당 셀을 전부 재호출합니다(원본은 보존, "
              f"중복 행은 로드 시 마지막 것만 집계).")
        return {}
    return progress


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
