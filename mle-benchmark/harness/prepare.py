#!/usr/bin/env python
"""Build K-MLE-Bench task packs from raw sources.

Writes, per task:
  kmle/packs/<task>/train.csv, test.csv, sample_submission.csv, meta.json
  kmle/private/<task>/answers.csv        <- never exposed to agents

Deterministic (SEED=42). Fresh splits from public train data (MLE-bench method):
hidden labels are never taken from a file published as an answer key.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]  # kmle/
RAW, PACKS, PRIVATE = ROOT / "raw", ROOT / "packs", ROOT / "private"


def emit(task, train, test, answers, sample, meta):
    pack, priv = PACKS / task, PRIVATE / task
    pack.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    train.to_csv(pack / "train.csv", index=False)
    test.to_csv(pack / "test.csv", index=False)
    sample.to_csv(pack / "sample_submission.csv", index=False)
    answers.to_csv(priv / "answers.csv", index=False)
    (pack / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[{task}] train={len(train)} test={len(test)} -> {pack}")


def t3_ynat():
    rows = json.loads((RAW / "ynat_train.json").read_text())
    df = pd.DataFrame([{"id": r["guid"], "title": r["title"], "label": r["label"]} for r in rows])
    classes = sorted(df.label.unique().tolist())
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    sample = pd.DataFrame({"id": te.id, "label": df.label.mode()[0]})
    emit("t3_ynat", tr, te[["id", "title"]], te[["id", "label"]], sample, {
        "id": "t3_ynat", "title_ko": "뉴스 토픽 분류 (KLUE-YNAT)",
        "metric": "macro_f1", "higher_is_better": True,
        "id_col": "id", "pred_cols": ["label"], "classes": classes,
        "source": "KLUE-YNAT v1.1 (CC-BY-SA 4.0)", "bands": "provisional — calibrate at pilot",
    })


def t4_nsmc():
    df = pd.read_csv(RAW / "nsmc_train.txt", sep="\t", quoting=3, dtype={"id": str})
    df = df.dropna(subset=["document"]).reset_index(drop=True)
    df["label"] = df.label.astype(int)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    sample = pd.DataFrame({"id": te.id, "label": 0})
    emit("t4_nsmc", tr, te[["id", "document"]], te[["id", "label"]], sample, {
        "id": "t4_nsmc", "title_ko": "네이버 영화리뷰 감성 분석 (NSMC)",
        "metric": "accuracy", "higher_is_better": True,
        "id_col": "id", "pred_cols": ["label"], "classes": [0, 1],
        "source": "NSMC (e9t/nsmc, open)", "bands": "provisional — calibrate at pilot",
    })


def t5_bike():
    cols = ["date", "rented_bike_count", "hour", "temperature_c", "humidity_pct",
            "wind_speed_ms", "visibility_10m", "dew_point_c", "solar_radiation_mj",
            "rainfall_mm", "snowfall_cm", "seasons", "holiday", "functioning_day"]
    df = pd.read_csv(RAW / "seoul_bike.csv", encoding="latin1")
    df.columns = cols
    dt = pd.to_datetime(df.date, format="%d/%m/%Y")
    df.insert(0, "id", dt.dt.strftime("%Y%m%d") + "_" + df.hour.astype(str).str.zfill(2))
    df = df.assign(_dt=dt).sort_values(["_dt", "hour"]).drop(columns="_dt").reset_index(drop=True)
    dates = sorted(dt.dt.date.unique())
    cutoff = dates[int(len(dates) * 0.8)]
    is_test = pd.to_datetime(df.date, format="%d/%m/%Y").dt.date >= cutoff
    tr, te = df[~is_test], df[is_test]
    sample = pd.DataFrame({"id": te.id, "rented_bike_count": 0})
    emit("t5_bike", tr, te.drop(columns="rented_bike_count"),
         te[["id", "rented_bike_count"]], sample, {
        "id": "t5_bike", "title_ko": "서울시 공공자전거 수요 예측 (시간 단위)",
        "metric": "rmse", "higher_is_better": False,
        "id_col": "id", "pred_cols": ["rented_bike_count"],
        "split": f"chronological, hidden from {cutoff.isoformat()}",
        "source": "UCI Seoul Bike Sharing Demand (CC BY 4.0)", "bands": "provisional — calibrate at pilot",
    })


def t1_pubg():
    """Korean-origin (Krafton) tabular task. Subsampled by match to stay CPU-friendly;
    split is match-grouped so no matchId leaks across train/test."""
    src = RAW / "pubg" / "train_V2.csv"
    if not src.exists():
        print("[t1_pubg] SKIP — raw/pubg/train_V2.csv missing (accept Kaggle rules, re-download)")
        return
    df = pd.read_csv(src).dropna(subset=["winPlacePerc"])
    rng = np.random.RandomState(SEED)
    matches = np.sort(df.matchId.unique())
    keep = rng.choice(matches, size=3000, replace=False)
    df = df[df.matchId.isin(keep)].reset_index(drop=True)
    tr_m, te_m = train_test_split(np.sort(keep), test_size=0.2, random_state=SEED)
    tr = df[df.matchId.isin(tr_m)]
    te = df[df.matchId.isin(te_m)]
    sample = pd.DataFrame({"Id": te.Id, "winPlacePerc": 0.5})
    emit("t1_pubg", tr, te.drop(columns="winPlacePerc"),
         te[["Id", "winPlacePerc"]], sample, {
        "id": "t1_pubg", "title_ko": "배틀그라운드 최종 순위 예측 (PUBG)",
        "metric": "mae", "higher_is_better": False,
        "id_col": "Id", "pred_cols": ["winPlacePerc"],
        "split": "match-grouped 80/20 on a 3000-match subsample",
        "source": "Kaggle pubg-finish-placement-prediction (Krafton data)",
        "bands": "provisional — calibrate at pilot",
    })


def t2_spooky():
    src = RAW / "spooky" / "train.csv"
    if not src.exists():
        print("[t2_spooky] SKIP — raw/spooky/train.csv missing (accept Kaggle rules, re-download)")
        return
    df = pd.read_csv(src)  # id,text,author in {EAP,HPL,MWS}
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.author)
    sample = pd.DataFrame({"id": te.id, "EAP": 1 / 3, "HPL": 1 / 3, "MWS": 1 / 3})
    emit("t2_spooky", tr, te[["id", "text"]], te[["id", "author"]], sample, {
        "id": "t2_spooky", "title_ko": "괴기 소설 작가 판별 (Spooky Author Identification)",
        "metric": "multiclass_logloss", "higher_is_better": False,
        "id_col": "id", "pred_cols": ["EAP", "HPL", "MWS"], "truth_col": "author",
        "classes": ["EAP", "HPL", "MWS"],
        "source": "Kaggle spooky-author-identification (MLE-bench lite comp)", "bands": "provisional — calibrate at pilot",
    })


def t6_klue_nli():
    rows = json.loads((RAW / "klue_nli_train.json").read_text())
    df = pd.DataFrame([{"id": r["guid"], "premise": r["premise"],
                        "hypothesis": r["hypothesis"], "label": r["gold_label"]}
                       for r in rows])
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    sample = pd.DataFrame({"id": te.id, "label": "neutral"})
    emit("t6_klue_nli", tr, te[["id", "premise", "hypothesis"]], te[["id", "label"]],
         sample, {"id": "t6_klue_nli", "title_ko": "자연어 추론 (KLUE-NLI)",
                  "metric": "accuracy_str", "higher_is_better": True,
                  "id_col": "id", "pred_cols": ["label"],
                  "classes": ["entailment", "neutral", "contradiction"],
                  "source": "KLUE-NLI v1.1 (CC-BY-SA 4.0)",
                  "bands": "provisional — calibrate at pilot"})


def t7_klue_sts():
    rows = json.loads((RAW / "klue_sts_train.json").read_text())
    df = pd.DataFrame([{"id": r["guid"], "sentence1": r["sentence1"],
                        "sentence2": r["sentence2"], "score": r["labels"]["label"]}
                       for r in rows])
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    sample = pd.DataFrame({"id": te.id, "score": 2.5})
    emit("t7_klue_sts", tr, te[["id", "sentence1", "sentence2"]],
         te[["id", "score"]], sample,
         {"id": "t7_klue_sts", "title_ko": "문장 유사도 (KLUE-STS)",
          "metric": "pearson", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["score"],
          "source": "KLUE-STS v1.1 (CC-BY-SA 4.0)",
          "bands": "provisional — calibrate at pilot"})


def t8_beep():
    df = pd.read_csv(RAW / "beep_train.tsv", sep="\t", quoting=3)
    df = df.rename(columns={"comments": "comment", "hate": "label"})
    df["id"] = [f"beep_{i:06d}" for i in range(len(df))]
    df = df[["id", "comment", "label"]]
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    sample = pd.DataFrame({"id": te.id, "label": "none"})
    emit("t8_beep", tr, te[["id", "comment"]], te[["id", "label"]], sample,
         {"id": "t8_beep", "title_ko": "혐오 표현 분류 (BEEP!)",
          "metric": "macro_f1", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["label"],
          "classes": ["none", "offensive", "hate"],
          "source": "kocohub/korean-hate-speech (CC-BY-SA 4.0)",
          "bands": "provisional — calibrate at pilot"})


def t9_korquad():
    data = json.loads((RAW / "korquad_train.json").read_text())["data"]
    arts = sorted(range(len(data)), key=lambda i: data[i]["title"])
    tr_i, te_i = train_test_split(arts, test_size=0.2, random_state=SEED)
    def flatten(idx, with_answer):
        out = []
        for i in idx:
            for p in data[i]["paragraphs"]:
                for qa in p["qas"]:
                    row = {"id": qa["id"], "context": p["context"],
                           "question": qa["question"]}
                    if with_answer:
                        row["answer"] = qa["answers"][0]["text"]
                    out.append(row)
        return pd.DataFrame(out)
    tr, te = flatten(tr_i, True), flatten(te_i, True)
    sample = pd.DataFrame({"id": te.id, "answer": ""})
    emit("t9_korquad", tr, te[["id", "context", "question"]],
         te[["id", "answer"]], sample,
         {"id": "t9_korquad", "title_ko": "기계 독해 (KorQuAD 1.0)",
          "metric": "korquad", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["answer"],
          "split": "article-grouped 80/20",
          "source": "KorQuAD 1.0 (CC BY-ND 2.0 KR)",
          "bands": "provisional — calibrate at pilot"})


def t10_kornli():
    df = pd.read_csv(RAW / "kornli_train.tsv", sep="\t", quoting=3,
                     on_bad_lines="skip").dropna()
    df = df[df.gold_label.isin(["entailment", "neutral", "contradiction"])]
    df = df.sample(n=60000, random_state=SEED).reset_index(drop=True)
    df["id"] = [f"kornli_{i:06d}" for i in range(len(df))]
    df = df.rename(columns={"gold_label": "label"})[
        ["id", "sentence1", "sentence2", "label"]]
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.label)
    sample = pd.DataFrame({"id": te.id, "label": "neutral"})
    emit("t10_kornli", tr, te[["id", "sentence1", "sentence2"]],
         te[["id", "label"]], sample,
         {"id": "t10_kornli", "title_ko": "자연어 추론 (KorNLI/MultiNLI-ko)",
          "metric": "accuracy_str", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["label"],
          "classes": ["entailment", "neutral", "contradiction"],
          "source": "kakaobrain KorNLI (CC-BY-SA 4.0)",
          "bands": "provisional — calibrate at pilot"})


def t11_korsts():
    df = pd.read_csv(RAW / "korsts_train.tsv", sep="\t", quoting=3,
                     on_bad_lines="skip").dropna(subset=["sentence1", "sentence2", "score"])
    df["id"] = [f"korsts_{i:05d}" for i in range(len(df))]
    df = df[["id", "sentence1", "sentence2", "score"]]
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    sample = pd.DataFrame({"id": te.id, "score": 2.5})
    emit("t11_korsts", tr, te[["id", "sentence1", "sentence2"]],
         te[["id", "score"]], sample,
         {"id": "t11_korsts", "title_ko": "문장 유사도 (KorSTS)",
          "metric": "pearson", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["score"],
          "source": "kakaobrain KorSTS (CC-BY-SA 4.0)",
          "bands": "provisional — calibrate at pilot"})


def _emit_clf(tid, df, input_cols, meta_extra, int_label=True):
    """Shared emitter for single-label classification packs."""
    label = "label"
    strat = df[label] if df[label].nunique() < 100 else None
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=strat)
    maj = df[label].mode()[0]
    sample = pd.DataFrame({"id": te.id, label: maj})
    meta = {"id": tid, "id_col": "id", "pred_cols": [label],
            "higher_is_better": True, **meta_extra}
    emit(tid, tr[["id", *input_cols, label]], te[["id", *input_cols]],
         te[["id", label]], sample, meta)


def t12_kobest_boolq():
    df = pd.read_parquet(RAW / "kobest_boolq.parquet")
    df["id"] = [f"boolq_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)
    _emit_clf("t12_kobest_boolq", df, ["paragraph", "question"],
              {"title_ko": "지문 기반 참/거짓 판단 (KoBEST BoolQ)", "metric": "accuracy",
               "classes": [0, 1], "source": "KoBEST BoolQ (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t13_kobest_copa():
    df = pd.read_parquet(RAW / "kobest_copa.parquet")
    df["id"] = [f"copa_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)  # 0/1 → pick alternative_1 or _2
    _emit_clf("t13_kobest_copa", df,
              ["premise", "question", "alternative_1", "alternative_2"],
              {"title_ko": "인과 추론 (KoBEST COPA)", "metric": "accuracy",
               "classes": [0, 1], "source": "KoBEST COPA (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t14_kobest_wic():
    df = pd.read_parquet(RAW / "kobest_wic.parquet")
    df["id"] = [f"wic_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)
    _emit_clf("t14_kobest_wic", df, ["word", "context_1", "context_2"],
              {"title_ko": "문맥 내 동일 의미 판별 (KoBEST WiC)", "metric": "accuracy",
               "classes": [0, 1], "source": "KoBEST WiC (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t15_kobest_hellaswag():
    df = pd.read_parquet(RAW / "kobest_hellaswag.parquet")
    df["id"] = [f"hellaswag_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)  # 0..3
    _emit_clf("t15_kobest_hellaswag", df,
              ["context", "ending_1", "ending_2", "ending_3", "ending_4"],
              {"title_ko": "상황 이어짓기 추론 (KoBEST HellaSwag)", "metric": "accuracy",
               "classes": [0, 1, 2, 3], "source": "KoBEST HellaSwag (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t16_kobest_sentineg():
    df = pd.read_parquet(RAW / "kobest_sentineg.parquet")
    df["id"] = [f"sentineg_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)
    _emit_clf("t16_kobest_sentineg", df, ["sentence"],
              {"title_ko": "부정 표현 감성 분석 (KoBEST SentiNeg)", "metric": "accuracy",
               "classes": [0, 1], "source": "KoBEST SentiNeg (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t17_pawsx_ko():
    df = pd.read_parquet(RAW / "pawsx_ko.parquet").dropna(subset=["sentence1", "sentence2"])
    df["id"] = [f"pawsx_{i:05d}" for i in range(len(df))]
    df["label"] = df.label.astype(int)
    df = df.sample(n=30000, random_state=SEED).reset_index(drop=True)
    _emit_clf("t17_pawsx_ko", df, ["sentence1", "sentence2"],
              {"title_ko": "문장 의역 판별 (PAWS-X 한국어)", "metric": "accuracy",
               "classes": [0, 1], "source": "PAWS-X ko (CC-BY-2.0-style, Google)",
               "bands": "provisional"})


def t18_klue_re():
    rows = json.loads((RAW / "klue_re_train.json").read_text())
    df = pd.DataFrame([{"id": r["guid"], "sentence": r["sentence"],
                        "subject_entity": r["subject_entity"]["word"],
                        "object_entity": r["object_entity"]["word"],
                        "label": r["label"]} for r in rows])
    classes = sorted(df.label.unique().tolist())
    _emit_clf("t18_klue_re", df, ["sentence", "subject_entity", "object_entity"],
              {"title_ko": "관계 추출 (KLUE-RE)", "metric": "accuracy_str",
               "classes": classes, "source": "KLUE-RE v1.1 (CC-BY-SA 4.0)",
               "bands": "provisional"})


def t19_klue_mrc():
    data = json.loads((RAW / "klue_mrc_train.json").read_text())["data"]
    arts = sorted(range(len(data)), key=lambda i: i)
    tr_i, te_i = train_test_split(arts, test_size=0.2, random_state=SEED)
    def flat(idx):
        out = []
        for i in idx:
            for p in data[i]["paragraphs"]:
                for qa in p["qas"]:
                    ans = "" if qa.get("is_impossible") else (
                        qa["answers"][0]["text"] if qa["answers"] else "")
                    out.append({"id": qa["guid"], "context": p["context"],
                                "question": qa["question"], "answer": ans})
        return pd.DataFrame(out)
    tr, te = flat(tr_i), flat(te_i)
    tr, te = tr.sample(n=min(40000, len(tr)), random_state=SEED), te
    sample = pd.DataFrame({"id": te.id, "answer": ""})
    emit("t19_klue_mrc", tr, te[["id", "context", "question"]],
         te[["id", "answer"]], sample,
         {"id": "t19_klue_mrc", "title_ko": "기계 독해 (KLUE-MRC, 답 없음 포함)",
          "metric": "korquad", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["answer"],
          "split": "article-grouped; unanswerable → empty answer",
          "source": "KLUE-MRC v1.1 (CC-BY-SA 4.0)", "bands": "provisional"})


def _ner_entities(char_tags):
    ents, cur = [], None
    for ch, tag in char_tags:
        if tag.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = [ch, tag[2:]]
        elif tag.startswith("I-") and cur and tag[2:] == cur[1]:
            cur[0] += ch
        else:
            if cur:
                ents.append(cur)
                cur = None
    if cur:
        ents.append(cur)
    return [(t.strip(), lab) for t, lab in ents if t.strip()]


def t20_klue_ner():
    sents, cur = [], []
    for line in (RAW / "klue_ner_train.tsv").read_text().splitlines():
        if line.startswith("##"):
            continue
        if line == "":
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            cur.append((parts[0], parts[1]))
    if cur:
        sents.append(cur)
    rows = []
    for i, s in enumerate(sents):
        text = "".join(c for c, _ in s)
        ents = _ner_entities(s)
        gold = "|".join(f"{t}:{lab}" for t, lab in ents)
        rows.append({"id": f"ner_{i:05d}", "sentence": text, "entities": gold})
    df = pd.DataFrame(rows)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    sample = pd.DataFrame({"id": te.id, "entities": ""})
    emit("t20_klue_ner", tr, te[["id", "sentence"]], te[["id", "entities"]],
         sample, {"id": "t20_klue_ner", "title_ko": "개체명 인식 (KLUE-NER)",
                  "metric": "ner_f1", "higher_is_better": True,
                  "id_col": "id", "pred_cols": ["entities"],
                  "labels": ["PS", "LC", "OG", "DT", "TI", "QT"],
                  "source": "KLUE-NER v1.1 (CC-BY-SA 4.0)", "bands": "provisional"})


def t21_kmmlu():
    """Korean professional/academic knowledge, 4-way MC (KMMLU). label = correct
    option index 1..4 (1=A,2=B,3=C,4=D); modeled as a 4-way target over the
    question + four options."""
    df = pd.read_parquet(RAW / "kmmlu.parquet")
    df = df.sample(n=6000, random_state=SEED).reset_index(drop=True)
    df["id"] = [f"kmmlu_{i:05d}" for i in range(len(df))]
    df["label"] = df.answer.astype(int)
    _emit_clf("t21_kmmlu", df, ["question", "A", "B", "C", "D"],
              {"title_ko": "한국형 전문 지식 4지선다 (KMMLU)", "metric": "accuracy",
               "classes": [1, 2, 3, 4],
               "note": "label = 정답 보기 번호 (1=A, 2=B, 3=C, 4=D)",
               "source": "HAERAE-HUB/KMMLU (CC-BY-ND 4.0)", "bands": "provisional"})


# t22_haerae (HAE-RAE Bench) REMOVED 2026-08-01: the smoke showed agents solve
# Korean-trivia MC by reciting answers from parametric memory (GPT hand-authored an
# expert_predictions.csv → 0.945), bypassing ML engineering. Knowledge-recall tasks
# are a poor fit for an *engineering* benchmark; KMMLU (t21) is kept because it
# resisted recall (agents must actually model it). Task id t22 is intentionally
# left as a gap that documents this design lesson.


def t23_korfin_asc():
    """Aspect-based sentiment on Korean financial news (KorFin-ASC): given a
    sentence and a target entity (aspect), classify sentiment toward it."""
    df = pd.read_parquet(RAW / "korfin_asc.parquet")
    df = df.rename(columns={"SRC": "sentence", "ASPECT": "aspect",
                            "SENTIMENT": "label"})
    df["id"] = df.SID.astype(str)
    df["label"] = df.label.astype(str).str.strip()
    df = df[["id", "sentence", "aspect", "label"]]
    classes = sorted(df.label.unique().tolist())
    _emit_clf("t23_korfin_asc", df, ["sentence", "aspect"],
              {"title_ko": "금융 뉴스 속성 기반 감성 분석 (KorFin-ASC)",
               "metric": "macro_f1", "classes": classes,
               "source": "amphora/korfin-asc (CC-BY-SA 4.0)", "bands": "provisional"})


def t24_kor_unsmile():
    """Multi-label Korean toxicity (UnSmile): predict a fixed-order 10-bit
    multi-hot over hate categories (a comment may hit several at once)."""
    cats = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교",
            "기타 혐오", "악플/욕설", "clean"]
    df = pd.read_parquet(RAW / "kor_unsmile.parquet")
    df["id"] = [f"unsmile_{i:05d}" for i in range(len(df))]
    df["sentence"] = df["문장"].astype(str)
    df["labels"] = df["labels"].apply(lambda v: "".join(str(int(x)) for x in v))
    df = df.sample(n=9000, random_state=SEED).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    sample = pd.DataFrame({"id": te.id, "labels": "0" * len(cats)})
    emit("t24_kor_unsmile", tr[["id", "sentence", "labels"]],
         te[["id", "sentence"]], te[["id", "labels"]], sample,
         {"id": "t24_kor_unsmile", "title_ko": "다중 레이블 혐오 표현 분류 (Korean UnSmile)",
          "metric": "multilabel_f1", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["labels"], "classes": cats,
          "note": ("labels = 고정 순서 10비트 멀티핫 문자열 (" + ", ".join(cats) + ")"),
          "source": "smilegate-ai/kor_unsmile (license unspecified; research use)",
          "bands": "provisional"})


def t25_klue_dp():
    """Dependency parsing (KLUE-DP): for each token predict its head (1-indexed
    word position, 0=root) and dependency relation. Scored by LAS."""
    df = pd.read_parquet(RAW / "klue_dp.parquet")
    deprels = sorted(set(x for row in df["deprel"] for x in row))
    df = df.sample(n=6000, random_state=SEED).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        wf, heads, deps = list(r["word_form"]), list(r["head"]), list(r["deprel"])
        n = min(len(wf), len(heads), len(deps))
        parse = "|".join(f"{int(heads[j])}:{deps[j]}" for j in range(n))
        rows.append({"id": f"dp_{i:05d}", "sentence": r["sentence"],
                     "tokens": json.dumps(wf[:n], ensure_ascii=False), "parse": parse})
    out = pd.DataFrame(rows)
    tr, te = train_test_split(out, test_size=0.2, random_state=SEED)
    sample = pd.DataFrame({"id": te.id, "parse": ""})
    emit("t25_klue_dp", tr[["id", "sentence", "tokens", "parse"]],
         te[["id", "sentence", "tokens"]], te[["id", "parse"]], sample,
         {"id": "t25_klue_dp", "title_ko": "의존 구문 분석 (KLUE-DP)",
          "metric": "las", "higher_is_better": True,
          "id_col": "id", "pred_cols": ["parse"], "labels": deprels,
          "format": ("pipe로 구분된 토큰별 'head:deprel' (tokens 순서대로). "
                     "head = 1-indexed 어절 위치, 0=root"),
          "source": "KLUE-DP v1.1 (CC-BY-SA 4.0)", "bands": "provisional"})


BUILDERS = {"t1_pubg": t1_pubg, "t2_spooky": t2_spooky, "t3_ynat": t3_ynat,
            "t4_nsmc": t4_nsmc, "t5_bike": t5_bike, "t6_klue_nli": t6_klue_nli,
            "t7_klue_sts": t7_klue_sts, "t8_beep": t8_beep,
            "t9_korquad": t9_korquad, "t10_kornli": t10_kornli,
            "t11_korsts": t11_korsts, "t12_kobest_boolq": t12_kobest_boolq,
            "t13_kobest_copa": t13_kobest_copa, "t14_kobest_wic": t14_kobest_wic,
            "t15_kobest_hellaswag": t15_kobest_hellaswag,
            "t16_kobest_sentineg": t16_kobest_sentineg,
            "t17_pawsx_ko": t17_pawsx_ko, "t18_klue_re": t18_klue_re,
            "t19_klue_mrc": t19_klue_mrc, "t20_klue_ner": t20_klue_ner,
            "t21_kmmlu": t21_kmmlu,  # t22_haerae removed (recall exploit)
            "t23_korfin_asc": t23_korfin_asc, "t24_kor_unsmile": t24_kor_unsmile,
            "t25_klue_dp": t25_klue_dp}

if __name__ == "__main__":
    np.random.seed(SEED)
    wanted = sys.argv[1:] or list(BUILDERS)
    for name in wanted:
        BUILDERS[name]()
