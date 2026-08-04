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

    def add(self, parsed: Any, sample: Any) -> None:
        self._total += 1
        v = self.value_fn(parsed, sample)
        if v is not None:
            self._sum += float(v)
            self._valid += 1

    def finalize(self) -> dict[str, Any]:
        n = self._total if self.count_all else self._valid
        mean = self._sum / self._valid if self._valid else 0.0
        out: dict[str, Any] = {self.out_key: float(mean), "n_evaluated": n}
        out.update(self.static)
        return out


class MultiMeanAccumulator:
    """여러 per-sample 스칼라를 동시에 평균(예: TXT-1의 token_f1 + exact_match).

    value_fns: {out_key: fn(parsed, sample) -> float}. 모든 키가 같은 분모(전체 개수)로 평균.
    무효(파싱 실패)는 각 fn이 0.0을 반환하도록 태스크가 정의(기존 score()가 그렇게 동작).
    """

    def __init__(self, value_fns: dict[str, Any], *, static: dict[str, Any] | None = None) -> None:
        self.value_fns = value_fns
        self.static = static or {}
        self._sums = {k: 0.0 for k in value_fns}
        self._n = 0

    def add(self, parsed: Any, sample: Any) -> None:
        self._n += 1
        for k, fn in self.value_fns.items():
            self._sums[k] += float(fn(parsed, sample))

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: (self._sums[k] / self._n if self._n else 0.0) for k in self.value_fns
        }
        out["n_evaluated"] = self._n
        out.update(self.static)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 이진 분류: tn/fp/fn/tp. sklearn binary_metrics와 수치 동일.
# ─────────────────────────────────────────────────────────────────────────────
class BinaryAccumulator:
    """이진(0/1) 분류. parsed가 None이면 무효(n_unparsed로 집계, 채점 제외).

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
        self.n_unparsed = 0
        self.gold0 = self.gold1 = 0   # class_balance(정답 기준)

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is None:
            self.n_unparsed += 1
            return
        gold = sample.reference
        pred = int(parsed)
        gold = int(gold)
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
    """

    def __init__(self) -> None:
        self.conf: dict[tuple[int, int], int] = {}
        self.n_unparsed = 0

    def add(self, parsed: Any, sample: Any) -> None:
        if parsed is None:
            self.n_unparsed += 1
            return
        key = (int(sample.reference), int(parsed))
        self.conf[key] = self.conf.get(key, 0) + 1

    def _metrics(self) -> tuple[float, float, int]:
        total = sum(self.conf.values())
        if total == 0:
            return 0.0, 0.0, 0
        labels = sorted({g for g, _ in self.conf} | {p for _, p in self.conf})
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
        }


# ─────────────────────────────────────────────────────────────────────────────
# 멀티라벨 집합: micro(누적 tp/fp/fn) + macro(per-sample p/r/f1 합). multilabel_prf와 동일.
# ─────────────────────────────────────────────────────────────────────────────
class MultilabelAccumulator:
    """예측 라벨 집합 vs 정답 라벨 집합. micro/macro precision·recall·f1.

    valid_fn(parsed, sample) -> bool: 유효 샘플만 채점(기본: parsed is not None and sample.reference).
    """

    def __init__(self, valid_fn=None) -> None:
        self.valid_fn = valid_fn or (lambda p, s: p is not None and bool(s.reference))
        self.tp = self.fp = self.fn = 0
        self.sum_p = self.sum_r = self.sum_f1 = 0.0
        self.n = 0

    def add(self, parsed: Any, sample: Any) -> None:
        if not self.valid_fn(parsed, sample):
            return
        pred: set = parsed
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
