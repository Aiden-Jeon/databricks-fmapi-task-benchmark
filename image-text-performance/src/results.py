"""벤치마크 결과 저장 및 메니페스트 관리.

SampleResult: 개별 샘플 실행 결과 (모델·태스크·샘플 별).
RunManifest: 벤치마크 실행 메타데이터 (구성·시점·코드 버전).

JSON 직렬화 → results/<run_id>/samples.jsonl + manifest.json
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SampleResult:
    """개별 샘플 실행 결과."""

    model_id: str                      # 모델 ID (opus, sol, glm 등)
    task_id: str                       # 태스크 ID (IMG-1, TXT-4 등)
    sample_id: int                     # 데이터셋 내 샘플 인덱스
    reasoning_mode: str                # minimal 또는 full
    prompt: str                        # 입력 프롬프트 (저장 위해 필요시 truncate 가능)
    model_output: str                  # 모델 응답 (평문)
    reference: Any                     # 정답 (형식은 태스크마다 다름: str, list, dict 등)
    request_id: str | None             # FMAPI request_id (시간·비용 조인 키)
    finish_reason: str | None          # stop, length, error 등
    usage: dict[str, Any]              # {prompt_tokens, completion_tokens, ...}
    latency_ms_local: float            # 클라이언트 측 벽시계 시간(ms)
    timestamp: str                     # ISO 8601 timestamp


def make_run_id(version_suffix: str = "", *, results_root: str | Path = "results") -> str:
    """타임스탬프 기반 run ID 생성. **기존 디렉터리와 충돌하지 않는 값**을 돌려준다.

    형식: YYYY-MM-DDTHH-MM[_version_suffix]
    예시: 2026-07-31T14-00 또는 2026-07-31T14-00_v1

    분 단위라 같은 분에 두 실행을 시작하면 같은 ID가 나오고, 두 run의 결과가 한 디렉터리에
    섞인다(scores.json을 서로 덮어써 수치가 뒤섞인다). 이미 존재하면 `-2`, `-3`…을 붙여
    피한다 — 초 단위로 바꾸지 않는 이유는 기존 run-id 형식(문서·README 링크·index)이
    분 단위로 고정돼 있어 호환을 깨지 않기 위해서다.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H-%M")
    base = f"{timestamp}_{version_suffix}" if version_suffix else timestamp

    root = Path(results_root)
    candidate = base
    n = 2
    while (root / candidate).exists() or (Path("reports") / candidate).exists():
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def git_commit() -> str | None:
    """현재 커밋 SHA를 얻는다 (short form).

    git이 없거나 repo가 아니면 None 반환.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@dataclass
class RunManifest:
    """벤치마크 실행 메타데이터 (재현성·감사용)."""

    run_id: str                        # 고유 run ID
    created_at: str                    # ISO 8601 타임스탬프
    models: list[dict[str, Any]]       # [{"id": "opus", "endpoint": "...", "family": "claude"}, ...]
    reasoning_modes: list[str]         # ["minimal", "full"]
    task_ids: list[str]                # ["IMG-1", "IMG-2", ..., "TXT-8"]
    git_commit: str | None             # 코드 버전 (short SHA)
    datasets: dict[str, Any] = field(default_factory=dict)   # 태스크별 데이터셋 id·split (재현성 §12)
    pricing: dict[str, Any] = field(default_factory=dict)    # usd_per_dbu 등 (비용 재현 §12)
    samples_per_task: int | None = None                       # 태스크당 샘플 수
    seed: int | None = None                                   # subset 고정 seed
    # 어느 Databricks 워크스페이스로 호출·과금됐는지. 프로파일이 다르면 엔드포인트 가용성·
    # 지연·단가가 달라져 run 간 비교의 전제가 바뀐다 → 재현성 메타에 포함.
    profile: str | None = None
    notes: str = ""                    # 추가 메모 (선택)

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화용 dict로 변환."""
        return asdict(self)


def append_sample_results(run_dir: str | Path, results: list[SampleResult]) -> Path:
    """샘플 결과를 samples.jsonl에 **append**한다(스트리밍 저장용).

    전체를 한 번에 truncate-쓰기 하는 대신, "a" 모드로 열어 이미 있는 줄을 보존하고
    새 줄만 이어 쓴다(각 셀 완료 시 호출). 매 호출마다 flush해, 도중에 프로세스가 죽어도
    이미 쓴 줄은 디스크에 남는다(crash-resume의 기반). run_dir이 없으면 생성.

    빈 리스트면 아무것도 쓰지 않고 경로만 돌려준다.

    Returns: samples.jsonl 파일 경로.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    samples_file = run_dir / "samples.jsonl"
    if not results:
        return samples_file

    with open(samples_file, "a", encoding="utf-8") as f:
        for result in results:
            # set(태그셋·키프레이즈 reference 등)은 JSON 미지원 → list로 변환
            line = json.dumps(asdict(result), ensure_ascii=False, default=_json_default)
            f.write(line + "\n")
        f.flush()

    return samples_file


def load_sample_results(run_dir: str | Path) -> list[SampleResult]:
    """samples.jsonl을 읽어 SampleResult 리스트로 복원(리포트 생성용, resume 검사용).

    - 파일이 없으면 빈 리스트.
    - JSON으로 저장하며 set→list 변환이 일어났으므로, 복원되는 reference는 list일 수 있다
      (리포트는 model_output·reference·finish_reason 등 스칼라/문자열만 읽으므로 무해).
    - SampleResult 필드 이외의 잉여 키는 방어적으로 무시(포맷 진화 대비).
    - 깨진 줄(부분 기록 등)은 조용히 건너뛴다.
    """
    run_dir = Path(run_dir)
    samples_file = run_dir / "samples.jsonl"
    if not samples_file.exists():
        return []

    valid_keys = {f.name for f in fields(SampleResult)}
    out: list[SampleResult] = []
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # 부분 기록·손상 줄 스킵
            if not isinstance(d, dict):
                continue
            try:
                out.append(SampleResult(**{k: v for k, v in d.items() if k in valid_keys}))
            except TypeError:
                continue  # 필드 누락 등 스키마 불일치 줄 스킵
    return out


def _json_default(o: Any) -> Any:
    """JSON 직렬화 fallback: set→정렬 list, 그 외는 str."""
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    return str(o)


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """메니페스트를 results/<run_id>/manifest.json에 저장.

    Returns: manifest.json 파일 경로.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

    return manifest_file
