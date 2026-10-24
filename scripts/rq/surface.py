"""RQ1 of docs/OPTIMIZER.md: how much of one run's AUROC do answer length and form explain?

Analyses A to D for a finished run, from its saved files only (no LLM, no GPU). Run it as
`run_pipeline.py --stage rq` with the same arguments as the run, so the file key, the answer
form and the train/test split are the run's own. One row per run is appended to results/rq.csv.

B replaces the length-matched subset (experiment F): instead of subsampling to equal counts,
it compares only (correct, hallucinated) pairs whose answers have the same number of words,
so every test row whose word count occurs with both labels is used. Exact counts, not the
coarse LENGTH_BINS of --length_match: inside a bin such as 9-12 or 13+ the hallucinated answers
are still longer, and that leftover length would leak into the "length held equal" AUC.
"""
import csv
import os
from datetime import datetime

import numpy as np
from scipy.stats import chi2, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.databuild.checks import answer_shape
from src.data.feature_extractor import SEQ_DIM
from src.data.loader import FINAL_PUNCT

RQ_CSV = "results/rq.csv"
N_BOOT = 1000
# ponytail: answers of MAX_WORDS words or more share one stratum; few correct answers are that long.
MAX_WORDS = 20


def word_strata(answers):
    """Stratum of each answer for B: its word count, capped at MAX_WORDS."""
    return np.minimum([len(a.split()) for a in answers], MAX_WORDS)


def surface_dense(answers):
    """(a) length: log words, log characters; (b) form: final . ! ? or danda, first letter
    upper case, all lower case, has a digit. One row per answer."""
    rows = []
    for a in (a.strip() for a in answers):
        rows.append([np.log1p(len(a.split())), np.log1p(len(a)), a != "" and a[-1] in FINAL_PUNCT,
                     a[:1].isupper(), a == a.lower(), any(c.isdigit() for c in a)])
    return np.array(rows, dtype=float)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def _shape_model():
    """Shape n-grams of the answer, as `build_data.py leak`: form without word content."""
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False),
                         LogisticRegression(max_iter=2000))


def stratified_auc(y, score, strata):
    """AUC over the (correct, hallucinated) pairs of the same stratum only: the per-stratum AUCs
    weighted by their number of pairs. Strata with one label only drop out; nan if all do."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1 = int(y[m].sum())
        pairs = n1 * (int(m.sum()) - n1)
        if pairs:
            num += roc_auc_score(y[m], score[m]) * pairs
            den += pairs
    return num / den if den else float("nan")


def per_stratum_auc(y, score, strata):
    """{word count: (AUC, n correct, n hallucinated)} for the strata with both labels."""
    out = {}
    for s in np.unique(strata):
        m = strata == s
        n1 = int(y[m].sum())
        if 0 < n1 < m.sum():
            name = f"{s}+" if s == MAX_WORDS else str(s)
            out[name] = (roc_auc_score(y[m], score[m]), int(m.sum()) - n1, n1)
    return out


def bootstrap_indices(y, strata, seed, n_boot=N_BOOT):
    """Resample rows inside every (stratum, label) cell, so each bootstrap set keeps the counts
    that weight the stratified AUC."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero((strata == s) & (y == c)) for s in np.unique(strata) for c in (0, 1)]
    cells = [c for c in cells if len(c)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(c, len(c)) for c in cells])


def ci95(values):
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _loglik(X, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000)).fit(X, y)
    return -log_loss(y, model.predict_proba(X)[:, 1], normalize=False)


def added_value(S, h, y, seed):
    """C: cross-fitted `label ~ surface` against `label ~ surface + hidden score` on the test rows.
    Returns both out-of-fold probability vectors and the likelihood-ratio p-value (1 df)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    Sh = np.column_stack([S, h])
    p_s = cross_val_predict(_lr(), S, y, cv=cv, method="predict_proba")[:, 1]
    p_sh = cross_val_predict(_lr(), Sh, y, cv=cv, method="predict_proba")[:, 1]
    lr_stat = max(0.0, 2 * (_loglik(Sh, y) - _loglik(S, y)))
    return p_s, p_sh, float(chi2.sf(lr_stat, 1))


def form_in_features(X_seq, X_glob, answers, seed):
    """D: can the stage 1 features predict the form? Cross-validated R² of a ridge regression to
    log word count, and AUC of a logistic regression to "ends with . ! ? or danda" (nan when
    almost every answer has the same ending, as with --answer_form)."""
    X = np.column_stack([X_seq[:, :, :SEQ_DIM].reshape(len(X_seq), -1), X_glob])
    dense = surface_dense(answers)
    r2 = cross_val_score(make_pipeline(StandardScaler(), Ridge(1.0)), X, dense[:, 0], cv=5, scoring="r2").mean()
    punct = dense[:, 2].astype(int)
    punct_auc = float("nan")
    if 5 <= punct.sum() <= len(punct) - 5:
        punct_auc = cross_val_score(_lr(), X, punct, scoring="roc_auc",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
    return float(r2), float(punct_auc)


def verdict(share, strat_auc, gain_lo):
    """Decision rule of RQ1, with the stratified AUC in place of the matched-subset AUC."""
    if share >= 0.9 and strat_auc <= 0.75:
        return "mostly form"
    if strat_auc >= 0.85 and gain_lo > 0:
        return "own signal"
    return "in between"


def analyse_run(samples, feat_key, key, seed, test_size, run_info, out_csv=RQ_CSV):
    """A to D for one run. `samples` are the rows as the LLM read them (answer form applied)."""
    y = np.load(f"results/features/y_{feat_key}.npy")
    if [s["is_hallucination"] for s in samples] != y.tolist():
        raise ValueError("rq: the features do not line up with the data file row by row "
                         "(rows lost in stage 1, or another --seed for TriviaQA); re-run stage 1.")
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    y_te = np.load(f"results/oof/y_test_{key}.npy")
    if not np.array_equal(y[te], y_te):
        raise ValueError(f"rq: the rebuilt split does not match results/oof/y_test_{key}.npy; "
                         "pass the --seed and --tag of the run.")
    y_tr = y[tr]
    answers = [s["answer"] for s in samples]
    dense = surface_dense(answers)
    shapes = np.array([answer_shape(a) for a in answers], dtype=object)
    strata = word_strata(answers)[te]

    hidden = np.load(f"results/oof/test_logits_{key}.npy").mean(axis=0)
    probs_path = f"results/oof/test_probs_{key}.npy"
    probs = np.load(probs_path) if os.path.exists(probs_path) else None
    reported = probs if probs is not None else hidden

    # A: surface-only models fitted on the run's training rows, scored on its test rows.
    surf = {
        "length": _lr().fit(dense[tr][:, :2], y_tr).predict_proba(dense[te][:, :2])[:, 1],
        "form": _lr().fit(dense[tr][:, 2:], y_tr).predict_proba(dense[te][:, 2:])[:, 1],
        "shape": _shape_model().fit(shapes[tr], y_tr).predict_proba(shapes[te])[:, 1],
    }

    # C: surface design = the dense features plus the shape model's logit, all on test rows.
    p = np.clip(surf["shape"], 1e-6, 1 - 1e-6)
    S = np.column_stack([dense[te], np.log(p / (1 - p))])
    p_s, p_sh, lr_p = added_value(S, hidden, y_te, seed)

    # B and C intervals from the same stratified bootstrap sets.
    # strat_surface is the control for B: the surface-only score of C held to the same word
    # count. Close to B_strat_reported means the "equal length" signal is still form.
    boot = {"strat_hidden": [], "strat_reported": [], "strat_surface": [], "gain": []}
    for b in bootstrap_indices(y_te, strata, seed):
        yb, sb = y_te[b], strata[b]
        boot["strat_hidden"].append(stratified_auc(yb, hidden[b], sb))
        boot["strat_reported"].append(stratified_auc(yb, reported[b], sb))
        boot["strat_surface"].append(stratified_auc(yb, p_s[b], sb))
        boot["gain"].append(roc_auc_score(yb, p_sh[b]) - roc_auc_score(yb, p_s[b]))

    auc_reported = roc_auc_score(y_te, reported)
    auc_surface = roc_auc_score(y_te, p_s)
    share = (auc_surface - 0.5) / (auc_reported - 0.5)
    strat_reported = stratified_auc(y_te, reported, strata)
    gain_lo, gain_hi = ci95(boot["gain"])
    r2_words, punct_auc = form_in_features(np.load(f"results/features/X_seq_{feat_key}.npy"),
                                           np.load(f"results/features/X_glob_{feat_key}.npy"),
                                           answers, seed)
    rho = {c: spearmanr(hidden[y_te == c], dense[te][y_te == c, 0]).statistic for c in (0, 1)}
    bins = per_stratum_auc(y_te, hidden, strata)

    row = {
        "time": datetime.now().isoformat(timespec="seconds"), **run_info,
        "reported": "stacker" if probs is not None else "network logit",
        "n_test": len(y_te),
        "auc_reported": auc_reported,
        "auc_hidden": roc_auc_score(y_te, hidden),
        "A_length": roc_auc_score(y_te, surf["length"]),
        "A_form": roc_auc_score(y_te, surf["form"]),
        "A_shape": roc_auc_score(y_te, surf["shape"]),
        "B_strat_hidden": stratified_auc(y_te, hidden, strata),
        "B_strat_hidden_lo": ci95(boot["strat_hidden"])[0],
        "B_strat_hidden_hi": ci95(boot["strat_hidden"])[1],
        "B_strat_reported": strat_reported,
        "B_strat_reported_lo": ci95(boot["strat_reported"])[0],
        "B_strat_reported_hi": ci95(boot["strat_reported"])[1],
        "B_strat_surface": stratified_auc(y_te, p_s, strata),
        "B_strat_surface_lo": ci95(boot["strat_surface"])[0],
        "B_strat_surface_hi": ci95(boot["strat_surface"])[1],
        "B_words_hidden":" ".join(f"{k}:{a:.3f}({n0}/{n1})" for k, (a, n0, n1) in bins.items()),
        "C_auc_surface": auc_surface,
        "C_auc_surface_hidden": roc_auc_score(y_te, p_sh),
        "C_gain": roc_auc_score(y_te, p_sh) - auc_surface, "C_gain_lo": gain_lo, "C_gain_hi": gain_hi,
        "C_lr_p": lr_p,
        "C_share_explained": share,
        "D_r2_log_words": r2_words,
        "D_auc_final_punct": punct_auc,
        "D_rho_correct": float(rho[0]), "D_rho_halluc": float(rho[1]),
        "verdict": verdict(share, strat_reported, gain_lo),
    }
    row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
    _append(row, out_csv)
    print("\n".join(f"{k}: {v}" for k, v in row.items()))
    print(f"Appended the RQ1 row to {out_csv}")
    return row


def _append(row, path):
    """Append one row. A file written with other columns (an older version of this script) is
    rewritten with the union of the columns; old rows get empty cells for the new ones."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows, fields = [], list(row)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old = reader.fieldnames or []
            if old == fields:
                rows = None
            else:
                rows = list(reader)
                fields = old + [k for k in fields if k not in old]
    if rows is None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        return
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows + [row])
    os.replace(tmp, path)
