"""온라인(스트리밍) 채점 누적기 — 메모리 O(1) 보장.

기존 태스크의 `score(parsed_list, samples)`는 전체 샘플을 리스트로 들고 한 번에 집계한다
(메모리 O(N)). 대규모 샘플에서 OOM의 원인이 되므로, 러너의 핫패스는 이 누적기를 쓴다:
샘플 하나를 처리할 때마다 `add()`로 상태만 갱신하고 파싱 결과·이미지는 즉시 버린다.
마지막에 `finalize()`가 태스크의 `score()`와 **동일한 dict**를 돌려준다(리포트 호환).

누적 상태는 전부 상수 크기다:
- 평균류(token_f1/rouge/cell_f1/caption/judge): sum + count
- 이진 분류: tn/fp/fn/tp 4개 카운트
- 다중 클래스: (gold,pred) 혼동 카운트 (클래스 수에 비례, 샘플 수와 무관)
- 멀티라벨: 누적 tp/fp/fn(micro) + per-sample p/r/f1 합(macro)

BERTScore만 전체 배치가 필요해 스트리밍이 불가한데, 현재 torch 미설치로 'deferred'라
실측에 영향 없음(설계 결정: 엄격한 O(1)에서 배치 메트릭은 제외).

각 태스크는 `make_accumulator()`로 자신의 누적기를 돌려주고, 러너가 add/finalize를 구동한다.
`score()`는 테스트·동치 검증용으로 남는다(누적기 finalize와 bit-identical해야 함).
"""

from __future__ import annotations

from typing import Any, Protocol


class _CallFailed:
    """호출 자체가 실패했음을 나타내는 sentinel. 누적기가 **분모에서 제외**한다.

    왜 None과 구분하는가 (2026-08-06):
    - `parsed is None` = 모델이 응답은 했지만 **형식을 못 맞춰 파싱 실패** → 실제 능력 문제이므로
      **0점으로 채점**해야 한다. None을 전부 제외하면, 형식을 대부분 못 맞추는 새 모델이
      "성공한 일부"만으로 높은 점수를 받는다.
    - `parsed is CALL_FAILED` = HTTP 502·타임아웃 등으로 **응답을 받지 못함** → 모델 성능이
      아니라 인프라 문제이므로 채점에서 빼야 한다.
    두 경우를 None으로 합쳐 두면 어느 쪽인지 알 수 없어 위 두 요구가 충돌한다.
    """

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return "CALL_FAILED"

    def __bool__(self) -> bool:
        return False


CALL_FAILED = _CallFailed()


def is_call_failed(parsed: Any) -> bool:
    """호출 실패 sentinel인지. 태스크 코드가 파싱 실패(None)와 구분할 때 쓴다."""
    return parsed is CALL_FAILED


class Accumulator(Protocol):
    """온라인 누적기 인터페이스. add로 상태 갱신, finalize로 score()와 동일 dict 반환."""

    def add(self, parsed: Any, sample: Any) -> None: ...

    def finalize(self) -> dict[str, Any]: ...


# ─────────────────────────────────────────────────────────────────────────────
# 평균류: per-sample 스칼라(또는 스칼라 dict)의 평균. sum+count만 유지.
# ─────────────────────────────────────────────────────────────────────────────
class MeanAccumulator:
    """per-sample 값 하나를 평균낸다(예: token_f1, cell_f1, caption_token_f1).

    value_fn(parsed, sample) -> float | None. None이면 무효 샘플로 카운트만 하고 합 제외
    (태스크별로 무효 처리 규칙이 달라 include_invalid_as_zero로 조정).

    **parsed=None(=호출 실패)은 분모에서도 뺀다** (2026-08-05). count_all=True인 태스크
    (TXT-3·IMG-6: 빈 예측도 낮은 점수로 포함)에서 호출 실패가 0점으로 섞이면 엔드포인트
    장애가 성능처럼 보인다. `n_skipped`로 빠진 건수를 드러낸다.
    """

    def __init__(
        self,
        out_key: str,
        value_fn,
        *,
        count_all: bool = True,
        static: dict[str, Any] | None = None,
    ) -> None:
        self.out_key = out_key
        self.value_fn = value_fn
        self.count_all = count_all          # n_evaluated가 전체(파싱 실패 포함)인지 유효분만인지
        self.static = static or {}          # notes 등 상수 키
        self._sum = 0.0
        self._valid = 0
        self._total = 0
        self._skipped = 0

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is CALL_FAILED:
            # 호출 실패(인프라) — 채점 대상이 아니다(모델이 답을 낸 게 아니므로).
            self._skipped += 1
            return
        # parsed is None = 파싱 실패(모델이 형식을 못 맞춤) → 아래 value_fn이 0점 처리.
        self._total += 1
        v = self.value_fn(parsed, sample)
        if v is not None:
            self._sum += float(v)
            self._valid += 1

    def finalize(self) -> dict[str, Any]:
        n = self._total if self.count_all else self._valid
        mean = self._sum / self._valid if self._valid else 0.0
        out: dict[str, Any] = {self.out_key: float(mean), "n_evaluated": n}
        if self._skipped:
            out["n_skipped"] = self._skipped
        out.update(self.static)
        return out


class MultiMeanAccumulator:
    """여러 per-sample 스칼라를 동시에 평균(예: TXT-1의 token_f1 + exact_match).

    value_fns: {out_key: fn(parsed, sample) -> float}. 모든 키가 같은 분모(전체 개수)로 평균.
    무효(파싱 실패)는 각 fn이 0.0을 반환하도록 태스크가 정의(기존 score()가 그렇게 동작).

    **parsed=None은 채점에서 제외한다** (2026-08-05). 러너는 **호출 자체가 실패한** 샘플에
    None을 넘긴다(HTTP 502·타임아웃 등). 그걸 0점으로 세면 엔드포인트 장애가 모델 성능처럼
    보인다 — 실측: opus IMG-2가 502로 11/30 실패했을 때 micro_f1 0.671 vs 성공분만 0.786.
    `n_skipped`로 몇 건이 빠졌는지 드러내, 조용히 분모가 줄지 않게 한다.
    (모델이 응답은 했지만 형식을 못 맞춘 '파싱 실패'는 여전히 0점이다 — 그건 실제 능력 문제다.)
    """

    def __init__(
        self,
        value_fns: dict[str, Any],
        *,
        static: dict[str, Any] | None = None,
        dynamic: dict[str, Any] | None = None,
    ) -> None:
        self.value_fns = value_fns
        self.static = static or {}
        # dynamic: {out_key: 인자 없는 callable}. **finalize 시점에** 호출해 값을 얻는다.
        # 생성 시점에 확정되지 않는 값(예: 한국어 토큰화 백엔드는 tokenize(ko) 최초 호출
        # 때 정해진다)을 static에 넣으면 항상 초기값("unknown")이 박힌다.
        self.dynamic = dynamic or {}
        self._sums = {k: 0.0 for k in value_fns}
        self._n = 0
        self._skipped = 0

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is CALL_FAILED:
            self._skipped += 1   # 호출 실패만 분모에서 제외
            return
        # 파싱 실패(None)는 각 value_fn이 0.0을 돌려주므로 0점으로 채점된다(능력 문제).
        self._n += 1
        for k, fn in self.value_fns.items():
            self._sums[k] += float(fn(parsed, sample))

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: (self._sums[k] / self._n if self._n else 0.0) for k in self.value_fns
        }
        out["n_evaluated"] = self._n
        if self._skipped:
            out["n_skipped"] = self._skipped
        out.update(self.static)
        for k, fn in self.dynamic.items():
            try:
                out[k] = fn()
            except Exception:
                pass   # 부가 정보라 실패해도 채점 결과를 버리지 않는다
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 이진 분류: tn/fp/fn/tp. sklearn binary_metrics와 수치 동일.
# ─────────────────────────────────────────────────────────────────────────────
class BinaryAccumulator:
    """이진(0/1) 분류.

    - `CALL_FAILED`(호출 실패, 인프라) → 채점 제외(`n_skipped`)
    - `None`(파싱 실패, 능력 문제) → **오답으로 채점**하고 `n_unparsed`로도 보고

    **파싱 실패를 분모에서 빼면 안 된다** (2026-08-06 지적, 실측 재현):
    정답 1개 + 파싱 실패 29개가 `accuracy=1.0, f1=1.0`으로 나왔다. yes/no를 못 낸 것은
    실제 능력 문제이므로 오답으로 세야 한다. 오답 방향은 **정답의 반대**로 기록해
    (gold=1이면 fn, gold=0이면 fp) accuracy·F1·혼동행렬이 모두 일관되게 벌점을 받는다.

    finalize는 binary_metrics와 동일한 accuracy/f1/confusion_matrix + n_evaluated/n_unparsed를
    돌려주고, class_balance는 태스크가 넘긴 라벨 이름 매핑으로 구성한다.
    """

    def __init__(
        self,
        *,
        class_balance_keys: tuple[str, str] | None = None,   # (라벨0 이름, 라벨1 이름); None이면 class_balance 생략
        include_confusion: bool = True,
    ) -> None:
        self.class_balance_keys = class_balance_keys
        self.include_confusion = include_confusion
        self.tp = self.tn = self.fp = self.fn = 0
        self.n_unparsed = 0        # 모델이 응답했지만 형식 파싱 실패(=오답으로 채점됨)
        self.n_call_failed = 0     # 호출 자체 실패(=인프라 문제, 채점 제외) — 둘을 구분해 보고
        self.gold0 = self.gold1 = 0   # class_balance(정답 기준)

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is CALL_FAILED:
            self.n_call_failed += 1   # 호출 실패 — 파싱 실패(n_unparsed)와 구분
            return
        gold = int(sample.reference)
        if parsed is None:
            # 파싱 실패 = 능력 문제 → **오답으로 채점**한다(정답의 반대를 예측한 것으로 기록).
            self.n_unparsed += 1
            if gold == 1:
                self.gold1 += 1
                self.fn += 1      # 정답 1을 못 맞힘
            else:
                self.gold0 += 1
                self.fp += 1      # 정답 0을 못 맞힘
            return
        pred = int(parsed)
        if gold == 1:
            self.gold1 += 1
        else:
            self.gold0 += 1
        if pred == 1 and gold == 1:
            self.tp += 1
        elif pred == 0 and gold == 0:
            self.tn += 1
        elif pred == 1 and gold == 0:
            self.fp += 1
        else:  # pred == 0 and gold == 1
            self.fn += 1

    def finalize(self) -> dict[str, Any]:
        n = self.tp + self.tn + self.fp + self.fn
        acc = (self.tp + self.tn) / n if n else 0.0
        f1_denom = 2 * self.tp + self.fp + self.fn
        f1 = (2 * self.tp / f1_denom) if f1_denom else 0.0
        out: dict[str, Any] = {"accuracy": float(acc), "f1": float(f1)}
        if self.include_confusion:
            out["confusion_matrix"] = {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp}
        out["n_evaluated"] = n
        out["n_unparsed"] = self.n_unparsed
        if self.n_call_failed:
            out["n_skipped"] = self.n_call_failed   # 호출 실패(파싱 실패와 구분)
        if self.class_balance_keys:
            k0, k1 = self.class_balance_keys
            out["class_balance"] = {k0: self.gold0, k1: self.gold1}
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 다중 클래스: (gold,pred) 혼동 카운트. sklearn classification_metrics와 동일.
# ─────────────────────────────────────────────────────────────────────────────
class MulticlassAccumulator:
    """다중 클래스 분류. accuracy + macro_f1(등장 라벨 기준, sklearn average='macro' 동일).

    상태는 dict[(gold,pred)] -> count 로 클래스 수에만 비례(샘플 수 무관).

    **파싱 실패(None)는 오답으로 채점**한다(BinaryAccumulator와 같은 이유 — 분모에서 빼면
    라벨을 못 내는 모델이 성공분만으로 만점을 받는다). 예측 라벨을 알 수 없으므로
    `UNPARSED_LABEL`이라는 별도 라벨로 기록한다 — 어떤 정답과도 일치하지 않아 오답이 되고,
    혼동행렬에서 "형식 실패"가 눈에 보인다.
    """

    # 실제 클래스 라벨(0,1,2,…)과 겹치지 않는 음수 sentinel. 파싱 실패의 '예측값'으로 쓴다.
    UNPARSED_LABEL = -1

    def __init__(self) -> None:
        self.conf: dict[tuple[int, int], int] = {}
        self.n_unparsed = 0        # 파싱 실패(오답으로 채점됨)
        self.n_call_failed = 0     # 호출 실패(인프라 문제, 채점 제외) — 구분해 보고

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is CALL_FAILED:
            self.n_call_failed += 1
            return
        if parsed is None:
            # 파싱 실패 = 능력 문제 → 오답으로 채점(정답 라벨 × UNPARSED 예측).
            self.n_unparsed += 1
            key = (int(sample.reference), self.UNPARSED_LABEL)
            self.conf[key] = self.conf.get(key, 0) + 1
            return
        key = (int(sample.reference), int(parsed))
        self.conf[key] = self.conf.get(key, 0) + 1

    def _metrics(self) -> tuple[float, float, int]:
        total = sum(self.conf.values())
        if total == 0:
            return 0.0, 0.0, 0
        # macro 평균 대상 라벨에서 UNPARSED_LABEL은 제외한다 — 실제 클래스가 아니라
        # "형식 실패" 표식이므로, 라벨로 세면 macro_f1 분모가 부풀어 점수가 왜곡된다.
        # (오답 자체는 fn으로 잡히므로 벌점은 그대로 반영된다.)
        labels = sorted(
            ({g for g, _ in self.conf} | {p for _, p in self.conf}) - {self.UNPARSED_LABEL}
        )
        correct = sum(c for (g, p), c in self.conf.items() if g == p)
        acc = correct / total
        f1s = []
        for lab in labels:
            tp = self.conf.get((lab, lab), 0)
            fp = sum(c for (g, p), c in self.conf.items() if p == lab and g != lab)
            fn = sum(c for (g, p), c in self.conf.items() if g == lab and p != lab)
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom else 0.0)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return acc, macro_f1, total

    def finalize(self) -> dict[str, Any]:
        acc, macro_f1, n = self._metrics()
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "n_evaluated": n,
            "n_unparsed": self.n_unparsed,
            **({"n_skipped": self.n_call_failed} if self.n_call_failed else {}),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 멀티라벨 집합: micro(누적 tp/fp/fn) + macro(per-sample p/r/f1 합). multilabel_prf와 동일.
# ─────────────────────────────────────────────────────────────────────────────
class MultilabelAccumulator:
    """예측 라벨 집합 vs 정답 라벨 집합. micro/macro precision·recall·f1.

    valid_fn(parsed, sample) -> bool: 유효 샘플만 채점(기본: parsed is not None and sample.reference).

    **호출 실패(CALL_FAILED)는 valid_fn보다 먼저 걸러 분모에서 제외**한다. 태스크가
    `valid_fn=lambda p,s: True`처럼 넓게 열어 둔 경우(TXT-7)에도 sentinel이 채점에 들어가
    `CALL_FAILED & set` 예외가 나거나 0점으로 섞이는 것을 막는다.
    파싱 실패(None)의 처리는 valid_fn에 맡긴다 — 태스크마다 "빈 예측을 0점으로 셀지"가 다르다.
    """

    def __init__(self, valid_fn=None, normalize_fn=None) -> None:
        self.valid_fn = valid_fn or (lambda p, s: p is not None and bool(s.reference))
        # normalize_fn: 채점 전에 예측을 변환(예: 파싱 실패 None → 빈 집합으로 0점 채점).
        # 태스크가 "파싱 실패도 채점 대상"이라고 선언하는 방법이다.
        self.normalize_fn = normalize_fn or (lambda p: p)
        self.tp = self.fp = self.fn = 0
        self.sum_p = self.sum_r = self.sum_f1 = 0.0
        self.n = 0
        self.n_call_failed = 0

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is CALL_FAILED:
            self.n_call_failed += 1
            return
        if not self.valid_fn(parsed, sample):
            return
        pred: set = self.normalize_fn(parsed)
        gold: set = sample.reference
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) else 0.0
        self.sum_p += p
        self.sum_r += r
        self.sum_f1 += f1
        self.n += 1

    def finalize(self) -> dict[str, Any]:
        micro_p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        micro_r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
        macro_p = self.sum_p / self.n if self.n else 0.0
        macro_r = self.sum_r / self.n if self.n else 0.0
        macro_f1 = self.sum_f1 / self.n if self.n else 0.0
        return {
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "n_evaluated": self.n,
        }


# ─────────────────────────────────────────────────────────────────────────────
# per-language 래퍼: 언어별 하위 누적기 + 전체 누적기. 상태는 언어 수(2)에만 비례.
# ─────────────────────────────────────────────────────────────────────────────
class PerLanguageAccumulator:
    """전체 + 언어별로 하위 누적기를 각각 굴린다(TXT-5/6/7/8 등 한/영 병행).

    make_sub: () -> Accumulator. langs: 집계할 언어 목록.
    merge_fn(overall_finalized, {lang: sub_finalized}) -> 최종 dict (태스크별 출력 형태 조립).
    """

    def __init__(self, make_sub, merge_fn, langs=("en", "ko")) -> None:
        self.overall = make_sub()
        self.by_lang = {lang: make_sub() for lang in langs}
        self.merge_fn = merge_fn

    def add(self, parsed: Any, sample: Any) -> None:
        self.overall.add(parsed, sample)
        sub = self.by_lang.get(getattr(sample, "lang", "en"))
        if sub is not None:
            sub.add(parsed, sample)

    def finalize(self) -> dict[str, Any]:
        overall = self.overall.finalize()
        per_lang = {lang: sub.finalize() for lang, sub in self.by_lang.items()}
        return self.merge_fn(overall, per_lang)


# ─────────────────────────────────────────────────────────────────────────────
# judge 평균: 1–5 점수 스트리밍 평균 (+ 선택적 per-language). 러너의 judge 훅이 사용.
# ─────────────────────────────────────────────────────────────────────────────
class JudgeMeanAccumulator:
    """judge 1–5 점수의 스트리밍 평균. add(score, lang)로 갱신."""

    def __init__(self, langs=("en", "ko")) -> None:
        self._sum = 0.0
        self._n = 0
        self._lang_sum = {l: 0.0 for l in langs}
        self._lang_n = {l: 0 for l in langs}

    def add_score(self, score: float | None, lang: str = "en") -> None:
        if score is None:
            return
        self._sum += float(score)
        self._n += 1
        if lang in self._lang_sum:
            self._lang_sum[lang] += float(score)
            self._lang_n[lang] += 1

    def finalize(self) -> dict[str, Any]:
        mean = self._sum / self._n if self._n else 0.0
        per_lang = {
            l: {"mean": (self._lang_sum[l] / self._lang_n[l] if self._lang_n[l] else None),
                "n": self._lang_n[l]}
            for l in self._lang_sum
        }
        return {"judge_mean": round(float(mean), 4), "n_judged": self._n, "per_language": per_lang}
