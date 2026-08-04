"""Feature extraction utilities for KoBEST HellaSwag.

Builds pairwise feature vectors (one row per candidate ending) describing the
relationship between the context and each ending, then a model scores each
candidate.  The candidate with the highest score is selected as the predicted
label.

No external data / internet required; only uses numpy / pandas / sklearn.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------- Jamo decomposition (Korean syllable -> letters) ----------
_CHOSUNG = [chr(c) for c in range(0x1100, 0x1113)]
_JUNGSUNG = [chr(c) for c in range(0x1161, 0x1176)]
_JONGSUNG = [chr(c) for c in range(0x11A8, 0x11C3)]


def split_jamo(text):
    out = []
    for ch in text:
        if '가' <= ch <= '힣':
            code = ord(ch) - 0xAC00
            cho = code // 588
            jung = (code - cho * 588) // 28
            jong = code - cho * 588 - jung * 28
            out.append(_CHOSUNG[cho])
            out.append(_JUNGSUNG[jung])
            if jong:
                out.append(_JONGSUNG[jong - 1])
        else:
            out.append(ch)
    return ''.join(out)


def char_tokens(text, n=3):
    """n-gram character tokens on jamo string (works for Korean)."""
    j = split_jamo(text)
    j = re.sub(r'\s+', ' ', j.strip())
    if len(j) < n:
        return [j] if j else []
    return [j[i:i + n] for i in range(len(j) - n + 1)]


def word_tokens(text):
    """Simple whitespace + punctuation tokens."""
    return re.findall(r'[가-힣A-Za-z0-9]+', text.lower())


def ngram_tokens(text, n=2):
    toks = word_tokens(text)
    if len(toks) < n:
        return ['_'.join(toks)] if toks else []
    return ['_'.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


# ---------- Split into sentences / clauses ----------
_SENT_SPLIT = re.compile(r'[.!?。！？\n]+')


def split_sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts


# ---------- Overlap / similarity features ----------
def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def containment(a, b):
    """How much of b is contained in a."""
    sa, sb = set(a), set(b)
    if not sb:
        return 0.0
    return len(sa & sb) / len(sb)


def overlap_count(a, b):
    return len(set(a) & set(b))


def longest_common_prefix_tokens(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_subsequence_ratio(a, b):
    """Cheap LCS-ish ratio using set intersection of tokens (rough)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


# ---------- Subject / entity tracking ----------
def extract_subjects(text):
    """Heuristic subject extraction: tokens ending with common subject
    markers or capitalized words / pronouns."""
    toks = word_tokens(text)
    return set(toks)


# ---------- Main feature builder ----------
def build_pairwise_features(df, tfidf_char=None, tfidf_word=None,
                            tfidf_word2=None, tfidf_char2=None,
                            fit=False):
    """Return X (4N x F), ids (4N,), group_idx (N,).

    Each example contributes 4 rows (one per ending).  Features describe the
    fit between context and the ending: length ratios, lexical overlap, and
    cosine similarity in several TF-IDF spaces.
    """
    n = len(df)
    contexts = df['context'].astype(str).tolist()
    endings = {k: df[f'ending_{k}'].astype(str).tolist() for k in range(1, 5)}

    # Tokenize for overlap features
    ctx_words = [word_tokens(c) for c in contexts]
    end_words = {k: [word_tokens(e) for e in endings[k]] for k in range(1, 5)}
    ctx_word2 = [ngram_tokens(c, 2) for c in contexts]
    end_word2 = {k: [ngram_tokens(e, 2) for e in endings[k]] for k in range(1, 5)}
    ctx_chars = [char_tokens(c, 3) for c in contexts]
    end_chars = {k: [char_tokens(e, 3) for e in endings[k]] for k in range(1, 5)}
    ctx_char2 = [char_tokens(c, 2) for c in contexts]
    end_char2 = {k: [char_tokens(e, 2) for e in endings[k]] for k in range(1, 5)}

    # sentence segmentation of context
    ctx_sents = [split_sentences(c) for c in contexts]
    ctx_last_sent_words = [
        word_tokens(s[-1]) if s else [] for s in ctx_sents]
    ctx_last_sent_chars = [
        char_tokens(s[-1], 3) if s else [] for s in ctx_sents]
    ctx_last_sent_text = [s[-1] if s else '' for s in ctx_sents]
    # second-to-last sentence
    ctx_prev_sent_words = [
        word_tokens(s[-2]) if len(s) >= 2 else [] for s in ctx_sents]
    ctx_prev_sent_chars = [
        char_tokens(s[-2], 3) if len(s) >= 2 else [] for s in ctx_sents]
    ctx_prev_sent_text = [s[-2] if len(s) >= 2 else '' for s in ctx_sents]
    # first sentence
    ctx_first_sent_words = [
        word_tokens(s[0]) if s else [] for s in ctx_sents]

    rows = []
    ids_long = []
    group = []
    for i in range(n):
        cw = ctx_words[i]
        cw2 = ctx_word2[i]
        cc = ctx_chars[i]
        cc2 = ctx_char2[i]
        last_w = ctx_last_sent_words[i]
        last_c = ctx_last_sent_chars[i]
        prev_w = ctx_prev_sent_words[i]
        prev_c = ctx_prev_sent_chars[i]
        first_w = ctx_first_sent_words[i]
        for k in range(1, 5):
            ew = end_words[k][i]
            ew2 = end_word2[k][i]
            ec = end_chars[k][i]
            ec2 = end_char2[k][i]

            ctx_len = len(contexts[i])
            end_len = len(endings[k][i])
            ctx_word_cnt = len(cw)
            end_word_cnt = len(ew)

            feat = {
                'ending_idx': k - 1,
                'ctx_len': ctx_len,
                'end_len': end_len,
                'len_ratio': end_len / max(ctx_len, 1),
                'ctx_word_cnt': ctx_word_cnt,
                'end_word_cnt': end_word_cnt,
                'word_cnt_ratio': end_word_cnt / max(ctx_word_cnt, 1),
                # overlap - words
                'word_jaccard': jaccard(cw, ew),
                'word_containment_ctx_in_end': containment(ew, cw),
                'word_containment_end_in_ctx': containment(cw, ew),
                'word_intersect_count': overlap_count(cw, ew),
                'word_dice': common_subsequence_ratio(cw, ew),
                # overlap - bigrams
                'word2_jaccard': jaccard(cw2, ew2),
                'word2_intersect_count': overlap_count(cw2, ew2),
                'word2_dice': common_subsequence_ratio(cw2, ew2),
                # overlap - char jamo trigrams
                'char_jaccard': jaccard(cc, ec),
                'char_intersect_count': overlap_count(cc, ec),
                'char_containment_end_in_ctx': containment(cc, ec),
                'char_containment_ctx_in_end': containment(ec, cc),
                'char_dice': common_subsequence_ratio(cc, ec),
                # overlap - char jamo bigrams
                'char2_jaccard': jaccard(cc2, ec2),
                'char2_intersect_count': overlap_count(cc2, ec2),
                'char2_dice': common_subsequence_ratio(cc2, ec2),
                # last sentence vs ending overlap
                'last_word_jaccard': jaccard(last_w, ew),
                'last_word_containment_end_in_last': containment(last_w, ew),
                'last_word_containment_last_in_end': containment(ew, last_w),
                'last_word_intersect_count': overlap_count(last_w, ew),
                'last_char_jaccard': jaccard(last_c, ec),
                'last_char_intersect_count': overlap_count(last_c, ec),
                'last_char_dice': common_subsequence_ratio(last_c, ec),
                'last_char_containment_end_in_last': containment(last_c, ec),
                'last_char_containment_last_in_end': containment(ec, last_c),
                # previous (second-to-last) sentence vs ending
                'prev_word_jaccard': jaccard(prev_w, ew),
                'prev_word_intersect_count': overlap_count(prev_w, ew),
                'prev_char_jaccard': jaccard(prev_c, ec),
                'prev_char_intersect_count': overlap_count(prev_c, ec),
                # first sentence vs ending
                'first_word_jaccard': jaccard(first_w, ew),
                'first_word_intersect_count': overlap_count(first_w, ew),
                # subject continuity: does ending share tokens with last
                # sentence's first few words (subject)?
                'last_first_word_in_end': float(
                    last_w[0] in ew) if last_w else 0.0,
                # ending vs ending avg similarity (distinguishing outliers)
            }
            rows.append(feat)
            ids_long.append(df['id'].iloc[i] if 'id' in df else i)
            group.append(i)

    X_basic = pd.DataFrame(rows)

    # TF-IDF similarity features
    if fit:
        corpus = contexts + [e for k in range(1, 5) for e in endings[k]]
        tfidf_char = TfidfVectorizer(
            analyzer=lambda s: char_tokens(s, 3),
            token_pattern=None, min_df=2, sublinear_tf=True)
        tfidf_char.fit(corpus)
        tfidf_char2 = TfidfVectorizer(
            analyzer=lambda s: char_tokens(s, 2),
            token_pattern=None, min_df=2, sublinear_tf=True)
        tfidf_char2.fit(corpus)
        tfidf_word = TfidfVectorizer(
            analyzer=word_tokens, token_pattern=None, min_df=2,
            sublinear_tf=True, ngram_range=(1, 1))
        tfidf_word.fit(corpus)
        tfidf_word2 = TfidfVectorizer(
            analyzer=lambda s: ngram_tokens(s, 2), token_pattern=None,
            min_df=2, sublinear_tf=True)
        tfidf_word2.fit(corpus)

    ctx_char_mat = tfidf_char.transform(contexts)
    ctx_char2_mat = tfidf_char2.transform(contexts)
    ctx_word_mat = tfidf_word.transform(contexts)
    ctx_word2_mat = tfidf_word2.transform(contexts)
    # last sentence vectors
    last_sents = [s[-1] if s else '' for s in ctx_sents]
    prev_sents = [s[-2] if len(s) >= 2 else '' for s in ctx_sents]
    last_char_mat = tfidf_char.transform(last_sents)
    last_word_mat = tfidf_word.transform(last_sents)
    last_word2_mat = tfidf_word2.transform(last_sents)
    prev_char_mat = tfidf_char.transform(prev_sents)
    prev_word_mat = tfidf_word.transform(prev_sents)

    from sklearn.metrics.pairwise import cosine_similarity

    sim_rows = []
    for i in range(n):
        cc_vec = ctx_char_mat[i]
        cc2_vec = ctx_char2_mat[i]
        cw_vec = ctx_word_mat[i]
        cw2_vec = ctx_word2_mat[i]
        lc_vec = last_char_mat[i]
        lw_vec = last_word_mat[i]
        lw2_vec = last_word2_mat[i]
        pc_vec = prev_char_mat[i]
        pw_vec = prev_word_mat[i]
        for k in range(1, 5):
            ev_char = tfidf_char.transform([endings[k][i]])
            ev_char2 = tfidf_char2.transform([endings[k][i]])
            ev_word = tfidf_word.transform([endings[k][i]])
            ev_word2 = tfidf_word2.transform([endings[k][i]])
            sim_char = float(cosine_similarity(cc_vec, ev_char)[0, 0])
            sim_char2 = float(cosine_similarity(cc2_vec, ev_char2)[0, 0])
            sim_word = float(cosine_similarity(cw_vec, ev_word)[0, 0])
            sim_word2 = float(cosine_similarity(cw2_vec, ev_word2)[0, 0])
            sim_last_char = float(cosine_similarity(lc_vec, ev_char)[0, 0])
            sim_last_word = float(cosine_similarity(lw_vec, ev_word)[0, 0])
            sim_last_word2 = float(cosine_similarity(lw2_vec, ev_word2)[0, 0])
            sim_prev_char = float(cosine_similarity(pc_vec, ev_char)[0, 0])
            sim_prev_word = float(cosine_similarity(pw_vec, ev_word)[0, 0])
            sim_rows.append({
                'sim_char': sim_char,
                'sim_char2': sim_char2,
                'sim_word': sim_word,
                'sim_word2': sim_word2,
                'sim_last_char': sim_last_char,
                'sim_last_word': sim_last_word,
                'sim_last_word2': sim_last_word2,
                'sim_prev_char': sim_prev_char,
                'sim_prev_word': sim_prev_word,
            })
    sim_df = pd.DataFrame(sim_rows).reset_index(drop=True)

    X = pd.concat([X_basic.reset_index(drop=True), sim_df], axis=1)
    return X, ids_long, np.array(group), tfidf_char, tfidf_word, tfidf_word2, tfidf_char2


def labels_to_long(df):
    """Expand per-example label to per-(example,candidate) target: 1 for the
    correct candidate, 0 otherwise. Returns y_long array (4N,)."""
    n = len(df)
    y_long = np.zeros(4 * n, dtype=np.float64)
    labels = df['label'].values if 'label' in df.columns else None
    if labels is None:
        return None
    for i in range(n):
        y_long[4 * i + int(labels[i])] = 1.0
    return y_long


def preds_long_to_labels(pred_long, group=None):
    """Given per-candidate scores (4N,), argmax within each group -> label (0-3).

    Assumes candidates for each question are contiguous in blocks of 4 and in
    ending_idx order 0..3.
    """
    pred_long = np.asarray(pred_long)
    n = pred_long.shape[0] // 4
    scores = pred_long.reshape(n, 4)
    return scores.argmax(axis=1)


if __name__ == '__main__':
    df = pd.read_csv('/tmp/kmle/M3_t15_kobest_hellaswag_full_20260804_033605/task/train.csv')
    X, ids, g, t1, t2, t3, t4 = build_pairwise_features(df, fit=True)
    print(X.shape, X.columns.tolist())
    print(X.head())
