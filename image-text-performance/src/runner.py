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
import sys
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

    args = parser.parse_args()

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

    # Run ID 및 디렉터리 생성
    run_id = make_run_id()
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Manifest 작성 (dry-run/real 공통)
    all_tasks = tasks_cfg.get("image_tasks", []) + tasks_cfg.get("text_tasks", [])
    task_ids = [t["id"] for t in all_tasks]

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

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        models=model_list,
        reasoning_modes=reasoning_override or models_cfg.reasoning_modes,
        task_ids=task_ids,
        git_commit=git_commit(),
        notes=f"Phase 0 dry-run" if args.dry_run else "Phase 0 skeleton (real execution not yet implemented)",
    )

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

    # 실제 실행 경로 (Phase 1 구현 예정)
    print("실제 샘플 루프 시작...")
    print("=" * 70)

    # FMAPIClient 생성 (나중에 사용할 예정)
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
        # TODO: Phase 1
        # - 데이터셋 로더 (datasets/registry.yaml에서 구성)
        # - 태스크 플러그인 (src/tasks/<id>.py에서 동적 로드)
        # - per-sample 루프: (model, task, reasoning_mode, sample) → FMAPI 호출 → SampleResult 저장
        # - 각 호출의 request_id 기록 → 나중에 cost/latency 조인용
        raise NotImplementedError(
            "실제 샘플 루프는 Phase 1에서 구현 예정. "
            "데이터셋 로더·태스크 플러그인·FMAPI 호출 루프 필요."
        )

    finally:
        fmapi.close()


if __name__ == "__main__":
    sys.exit(main())
