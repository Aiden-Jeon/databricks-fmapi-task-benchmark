"""
LLM-as-judge scoring orchestration.

This module handles LLM judge inference and score parsing. The judge (Databricks
Gemini-3.1-Pro) evaluates free-form generation tasks using structured rubrics
and produces 1–5 scores.

Key components:
- JUDGE_MAX_TOKENS: 모든 judge 호출이 공유하는 max_tokens. 아래 "잘림" 주석 참고.
- parse_judge_score: 판정 텍스트에서 1–5 점수를 추출. **없으면 None**(지어내지 않음).
- run_judge: judge 1회 호출 + 파싱. 실패는 None(전 태스크가 공유하는 단일 경로).
- summarize_judge_scores: None(실패) 제외 평균 + 실패 건수.
- load_rubrics: Load task-specific scoring rubrics from YAML config.
- build_judge_prompt: Compose a scoring prompt with anchored rubric to
  mitigate position and verbosity bias.

See plan.md appendix "채점 방법론 상세" (P0) for judge parsing strategy and
bias mitigation (position/verbosity).
"""

import re
from typing import Optional

import yaml

# judge(gemini)는 reasoning을 완전히 끌 수 없어 사고 토큰을 먼저 소비한다. max_tokens가
# 작으면 사고에 다 쓰고 본문이 잘려("finish_reason: length") 점수를 못 뱉는다.
# 실측(2026-08-05, IMG-1 캡션 판정): 256 → reasoning 240 / completion 12 / 문장 중간 잘림 →
# 파싱 실패. 1024 → finish=stop, 정상 "Score: 5". 짧은 판정(TXT-4 등)은 256에서도 통과해
# **길이에 따라 조용히 갈리므로**, 모든 태스크가 이 상수를 공유한다(태스크별 하드코딩 금지).
JUDGE_MAX_TOKENS = 1024


def parse_judge_score(text: str) -> Optional[int]:
    """
    judge 응답에서 1–5 점수를 추출한다. **명시적 점수 표현만 인정**한다.

    허용 형태(프롬프트가 "Score: 4"를 요구하므로 이 계열이 정상 경로):
    - "Score: 4", "score is 4", "Rating: 3", "**Final score:** 5"
    - "4/5", "4 out of 5"
    - 응답 전체가 숫자 하나뿐인 경우("4")
    [1, 5] 밖의 값은 절단한다(judge가 0이나 7을 말하는 드문 경우).

    **찾지 못하면 None을 돌려준다 — 숫자를 추측하지 않는다.**
    옛 구현은 폴백으로 (2) 아무 단독 1–5 숫자, (3) 아무 숫자나 주워 점수로 썼는데,
    judge 응답이 잘리면(gemini가 사고 토큰을 소진해 "finish_reason: length") 남은
    산문의 **본문 숫자**가 점수로 오인됐다. 실측된 오탐:
        "captures 1 of the 5 key elements"  → 5
        "The caption describes 2 sinks and 3 mirrors" → 3
        "anchor 4 description but"          → 4
    이 값들은 판정이 아니라 캡션 내용이라, 리포트의 judge 평균을 조용히 왜곡했다.
    파싱 실패는 None으로 드러내고 호출부가 집계에서 제외한다(3점 등으로 채우지 않는다).

    어댑터가 gemini의 content list를 이미 평문으로 정규화하므로 여기선 문자열만 다룬다.

    Args:
        text: judge 응답 평문.

    Returns:
        [1, 5] 정수, 또는 명시적 점수가 없으면 None.
    """
    if not text:
        return None

    text = text.strip()

    def _clamp(raw: str) -> Optional[int]:
        try:
            return max(1, min(5, int(float(raw))))
        except (TypeError, ValueError):
            return None

    # 1) "score/rating[:| is] N" — 마지막 매치를 쓴다(루브릭 인용보다 최종 판정이 뒤에 온다).
    #    앞에 **/#/공백 등 마크다운 장식 허용. 범위를 벗어난 값(0, 7 등)도 여기서 잡아
    #    _clamp가 [1,5]로 조정한다 — judge가 명시한 점수는 버리지 않는다.
    named = re.findall(
        r"(?:score|rating|점수)\s*(?:is|=|:|：)?\s*\**\s*(\d+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if named:
        return _clamp(named[-1])

    # 2) "4/5" 또는 "4 out of 5" — 분모가 5인 경우만(비율 표기)
    ratio = re.findall(r"\b([0-5])\s*(?:/|out\s+of)\s*5\b", text, re.IGNORECASE)
    if ratio:
        return _clamp(ratio[-1])

    # 3) 응답이 사실상 숫자 하나뿐 (예: "4", "4." , "**5**")
    bare = re.fullmatch(r"[^\w]*([1-5])[^\w]*", text)
    if bare:
        return _clamp(bare.group(1))

    # 명시적 점수 없음 → 추측하지 않는다.
    return None


def run_judge(
    judge_client,
    judge_endpoint: str,
    judge_prompt: str,
    task_id: str,
    sample_id: object = "?",
) -> Optional[int]:
    """judge 1회 호출 + 점수 파싱. 실패는 **None**을 돌려주고 원인을 로그로 남긴다.

    태스크마다 이 블록을 복붙하다 보니 max_tokens와 실패 처리가 서로 어긋났다
    (IMG-1은 None, 나머지는 조용히 3점, TXT-5만 1024). 여기로 모아 한 곳만 고치면
    전 태스크에 적용되게 한다.

    **파싱 실패에 3점 같은 값을 채우지 않는다.** 실패를 중간값으로 메우면 judge 평균이
    그럴듯하게 오염되고(실측: sol TXT-2의 3점 비율 70%), 사후에 실패였는지 판정이었는지
    구분할 수 없다. None은 호출부가 집계에서 제외하고 리포트에 실패 수로 드러낸다.

    Returns:
        [1,5] 점수, 또는 호출·파싱 실패 시 None.
    """
    try:
        response = judge_client.chat(
            endpoint=judge_endpoint,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
        )
    except Exception as e:
        print(f"  [{task_id} judge 호출 실패] s{sample_id}: {type(e).__name__}: {e}")
        return None

    score = parse_judge_score(response.text)
    if score is None:
        # finish_reason=length면 잘림(max_tokens 부족), 그 외면 형식 이탈 — 구분되게 남긴다.
        print(
            f"  [{task_id} judge 파싱 실패] s{sample_id} "
            f"finish={response.finish_reason} 응답={response.text[:80]!r}"
        )
    return score


def run_judge_batch(
    judge_client,
    judge_endpoint: str,
    items: list,
    task_id: str,
    *,
    max_workers: int = 1,
) -> list[Optional[int]]:
    """여러 judge 호출을 **병렬**로 수행하고 점수를 **입력 순서 그대로** 돌려준다.

    judge는 태스크마다 샘플 수만큼 순차 호출돼(gemini ~2.8초+/건) 생성 태스크의 꼬리
    지연이 컸다. 호출을 겹쳐 실행하면 그 시간이 준다. judge 엔드포인트도 rate limit이
    넉넉하고(429 미관측) 어댑터 토큰 갱신이 스레드 세이프하므로 안전하다.

    Args:
        items: `(judge_prompt, sample_id)` 튜플 리스트. **judge_prompt가 None이면** 그
            슬롯은 호출 없이 None으로 채운다(태스크가 빈 예측 등을 건너뛰는 경우 — 예: IMG-1이
            빈 캡션을 judge 없이 None 처리하던 동작을 그대로 보존).
        max_workers: 최대 동시 호출 수(1이면 순차 — 기존 동작과 동일).

    Returns:
        `list[Optional[int]]` — items와 같은 길이·순서. 각 원소는 [1,5] 점수 또는 None(실패/스킵).
        **실패를 값으로 메우지 않는다**(run_judge 계약 그대로): 호출·파싱 실패는 None이고
        호출부(summarize_judge_scores)가 평균에서 제외한다.
    """
    def _one(item):
        prompt, sample_id = item
        if prompt is None:
            return None   # 태스크가 스킵하기로 한 슬롯(빈 예측 등) — 호출하지 않는다
        return run_judge(judge_client, judge_endpoint, prompt, task_id, sample_id)

    # run_judge는 내부에서 모든 예외를 잡아 None을 돌려주므로 _one은 예외를 던지지 않는다.
    # 그래도 map_concurrent 계약상 (값, 예외)로 오므로, 예기치 못한 예외는 None으로 처리해
    # "실패=None" 축을 유지한다(점수를 지어내지 않는다).
    from src.adapters.concurrency import map_concurrent

    out: list[Optional[int]] = []
    for val, exc in map_concurrent(_one, items, max_workers=max_workers):
        out.append(None if exc is not None else val)
    return out


def summarize_judge_scores(scores: list[Optional[int]]) -> dict:
    """judge 점수 리스트를 집계. **None(실패)은 평균에서 제외**하고 개수로 보고한다.

    반환 키는 러너의 `_normalize_judge`가 읽는 계약을 유지한다:
    judge_mean / judge_scores / n_judged (+ n_judge_failed).
    유효 점수가 하나도 없으면 judge_mean=None — 0.0으로 두면 "최악 판정"처럼 보여
    리포트에서 성능 저하로 오독된다(옛 IMG-1의 judge_mean=0.0이 그 사례).
    """
    valid = [s for s in scores if s is not None]
    return {
        "judge_mean": (sum(valid) / len(valid)) if valid else None,
        "judge_scores": scores,
        "n_judged": len(valid),
        "n_judge_failed": len(scores) - len(valid),
    }


def load_rubrics(path: str = "config/judge_rubrics.yaml") -> dict:
    """
    Load judge rubrics from YAML configuration.

    Rubrics define scoring anchors (1–5) for each task, guiding consistent
    judge evaluation. File format:

        task_id:
          name: "Task Name"
          description: "What this task measures"
          anchors:
            1: "Poor description (1 point)"
            2: "Fair description (2 points)"
            ...
            5: "Excellent description (5 points)"

    Args:
        path: Path to rubrics YAML file (relative to project root).

    Returns:
        Dictionary mapping task_id -> rubric definition (with name,
        description, anchors).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        rubrics = yaml.safe_load(f)
    return rubrics or {}


def build_judge_prompt(
    task_id: str,
    question: str,
    reference: str,
    candidate: str,
    rubric: dict,
) -> str:
    """
    Compose a judge scoring prompt with anchored rubric.

    Constructs a prompt that:
    1. States the task and goal clearly.
    2. Presents the reference (ground truth) and candidate output.
    3. Embeds the 1–5 rubric to anchor scoring.
    4. Asks for a score in the specified range.

    Position and verbosity bias mitigation:
    - Reference and candidate are shown in fixed order (not shuffled per-sample,
      since the judge model is single and consistent).
    - Rubric is embedded inline to avoid hidden biases from unstated assumptions.
    - Prompt avoids length or style cues that might bias scoring.

    Args:
        task_id: Task identifier (e.g., "IMG-1", "TXT-4").
        question: Question or context (e.g., "What is in this image?").
        reference: Ground truth / reference output.
        candidate: Model's candidate output to be scored.
        rubric: Rubric dict. Two schemas are accepted:
                - config/judge_rubrics.yaml: {task, language, scale}
                - 태스크 인라인 fallback: {name, description, anchors}
                where the anchor map is {1: "...", 2: "...", ..., 5: "..."}.

    Returns:
        Formatted prompt string ready to send to judge model.
    """
    # rubric 스키마 2종 모두 허용(YAML의 task/scale ↔ 인라인 fallback의 name/anchors).
    # 예전엔 name/anchors만 읽어 YAML 루브릭의 scale이 조용히 무시되고(빈 앵커) judge
    # 프롬프트에 채점 기준이 빠지는 버그가 있었다.
    rubric_name = rubric.get("name") or rubric.get("task") or task_id
    rubric_desc = rubric.get("description", "")
    anchors = rubric.get("anchors") or rubric.get("scale") or {}

    # Build anchor text (키가 int/str 섞여도 안전하게 문자열 기준 정렬)
    anchor_text = "\n".join(
        f"  {score}: {desc}" for score, desc in sorted(anchors.items(), key=lambda kv: str(kv[0]))
    )

    prompt = f"""You are an expert evaluator for the following task:

**Task:** {rubric_name}
**Description:** {rubric_desc}

**Question/Context:**
{question}

**Reference (ground truth):**
{reference}

**Candidate output to evaluate:**
{candidate}

**Scoring rubric (1–5):**
{anchor_text}

Please evaluate the candidate output against the reference and provide a score from 1 to 5.
Include brief reasoning, then state your final score clearly (e.g., "Score: 4").
"""
    return prompt
