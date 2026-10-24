"""RQ2 -- Phan lon AUROC cua bo phat hien dua tren hidden state tren HaluEval co the duoc giai thich
boi dac diem be mat cua cau tra loi (do dai, van phong) khong?

Chay qua run_pipeline.py (log tong nam o logs/rq2_<time>.log do run_pipeline ghi):

    python run_pipeline.py --rq 2                                    # mistral-7b, en, halueval, 5 seed
    python run_pipeline.py --rq 2 --models mistral-7b llama2-7b      # ca 2 model (van chi en)
    python run_pipeline.py --rq 2 --seed 1                           # chay thu 1 seed
    python run_pipeline.py --rq 2 --skip b3 hs --dry_run             # xem ke hoach, khong chay

Ket qua: results/rq2/halueval_<lang>_<model>/  va  results/rq2/summary_all.csv

Cac thanh phan
  B0  chi do dai (LR, GBM)                 B1  thong ke van phong thu cong (LR, GBM)
  B2  TF-IDF word/char + LR, che do answer-only / question-only / question+answer
  B3  XLM-R fine-tune chi tren answer (tuy chon, can GPU)
  hs  mo hinh hidden-state cua pipeline (goi lai stage 2/3/4 cua run_pipeline.py, CUNG split/seed)
Thi nghiem co che
  length-bin / length-matched AUROC, complementarity (stack be mat + hidden), residualize
  (loai thanh phan tuyen tinh cua be mat khoi feature hidden roi train lai), perturbation,
  importance cua B1 (permutation + SHAP neu co), n-gram quan trong nhat cua B2.
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
import time
import traceback
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# src/rq/rq2_run.py -> root. run_pipeline chay script voi cwd=root nhung sys.path[0] la src/rq.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# LogisticRegressionCV in sklearn>=1.8 prints a FutureWarning per fit; it would flood the combined log.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RESULTS = ROOT / "results" / "rq2"
DEFAULT_SEEDS = [1, 11, 29, 37, 43]
PERTURBATIONS = ["lowercase", "no_punct", "shuffle_words", "cut_first3", "keep_first3"]
SKIPPABLE = ["b2", "b3", "hs", "resid", "complement", "perturb", "importance"]

# ----------------------------------------------------------------------------------------------
# Surface features
# ----------------------------------------------------------------------------------------------
_WORD = re.compile(r"\w+", re.UNICODE)

HEDGES = {
    "en": ["maybe", "perhaps", "possibly", "probably", "might", "may", "could", "likely", "seems",
           "unclear", "not sure", "i think", "however", "although", "approximately", "about",
           "around", "sorry", "as an ai", "cannot", "not provided", "no information", "there is no"],
    "fr": ["peut-être", "probablement", "possiblement", "environ", "sans doute", "cependant",
           "il semble", "pourrait", "désolé", "en tant qu'ia"],
    "vi": ["có thể", "có lẽ", "có vẻ", "khoảng", "dường như", "tuy nhiên", "không chắc",
           "hình như", "xin lỗi"],
}
_HEDGE_RE = {k: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in v) + r")\b", re.IGNORECASE)
             for k, v in HEDGES.items()}

B0_COLS = ["ans_chars", "ans_words", "ans_tokens", "len_ratio_words"]
B1_EXTRA = ["n_punct", "n_comma", "n_period", "ends_period", "ends_other_punct", "n_sentences",
            "n_upper", "frac_upper", "n_digit", "frac_digit", "n_cap_words", "frac_cap_words",
            "ttr", "avg_word_len", "n_quote", "n_paren", "frac_nonascii", "n_hedge",
            "overlap_a_in_q", "overlap_q_in_a", "starts_upper", "q_chars", "q_words", "q_tokens"]
ALL_COLS = B0_COLS + B1_EXTRA
I_B0 = [ALL_COLS.index(c) for c in B0_COLS]
I_TOK = ALL_COLS.index("ans_tokens")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def make_token_counter(model_key: str):
    """Dem token bang tokenizer cua chinh LLM (dung la thu hidden state nhin thay)."""
    try:
        from transformers import AutoTokenizer
        from src.config import MODEL_REGISTRY
        tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY[model_key])

        def count(texts):
            ids = tok(list(texts), add_special_tokens=False)["input_ids"]
            return np.array([len(x) for x in ids])
        return count
    except Exception as e:  # noqa: BLE001
        print(f"[warn] khong nap duoc tokenizer cua {model_key} ({type(e).__name__}: {e}); "
              f"dung xap xi 1.3 x so tu", flush=True)
        return lambda texts: np.array([int(round(len(_WORD.findall(t)) * 1.3)) for t in texts])


def surface_features(questions, answers, lang, count_tokens) -> np.ndarray:
    hedge_re = _HEDGE_RE.get(lang)
    a_tok, q_tok = count_tokens(answers), count_tokens(questions)
    rows = []
    for q, a, at, qt in zip(questions, answers, a_tok, q_tok):
        aw, qw = _WORD.findall(a), _WORD.findall(q)
        al = [w.lower() for w in aw]
        a_set, q_set = set(al), {w.lower() for w in qw}
        n_chars, n_words = len(a), len(aw)
        n_upper = sum(c.isupper() for c in a)
        n_alpha = max(1, sum(c.isalpha() for c in a))
        n_digit = sum(c.isdigit() for c in a)
        caps = sum(1 for w in aw[1:] if w[:1].isupper())
        stripped = a.rstrip()
        rows.append([
            n_chars, n_words, int(at), n_words / (len(qw) + 1),
            sum(_is_punct(c) for c in a), a.count(","), a.count("."),
            int(stripped.endswith(".")), int(bool(stripped) and stripped[-1] != "." and _is_punct(stripped[-1])),
            max(1, len(re.findall(r"[.!?]+", a))),
            n_upper, n_upper / n_alpha, n_digit, n_digit / max(1, n_chars),
            caps, caps / max(1, n_words - 1),
            len(a_set) / max(1, n_words), float(np.mean([len(w) for w in aw])) if aw else 0.0,
            a.count('"') + a.count("'"), a.count("(") + a.count(")"),
            sum(ord(c) > 127 for c in a) / max(1, n_chars),
            len(hedge_re.findall(a)) if hedge_re else 0,
            len(a_set & q_set) / max(1, len(a_set)), len(a_set & q_set) / max(1, len(q_set)),
            int(a[:1].isupper()), len(q), len(qw), int(qt),
        ])
    return np.asarray(rows, dtype=np.float64)


def apply_perturbation(name: str, answers):
    out = []
    for i, a in enumerate(answers):
        w = a.split()
        if name == "lowercase":
            out.append(a.lower())
        elif name == "no_punct":
            out.append(re.sub(r"\s+", " ", "".join(" " if _is_punct(c) else c for c in a)).strip())
        elif name == "shuffle_words":
            random.Random(1000 + i).shuffle(w)
            out.append(" ".join(w))
        elif name == "cut_first3":
            out.append(" ".join(w[3:]) if len(w) > 3 else (w[-1] if w else a))
        elif name == "keep_first3":
            out.append(" ".join(w[:3]))
        else:
            raise ValueError(name)
    return out


# ----------------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def fast_auc(y, s) -> float:
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_boot(y, a, b, n=1000, seed=0):
    """Bootstrap theo cap tren cung tap test: mean, 2.5%, 97.5% cua AUC(a) - AUC(b)."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(fast_auc(y[i], a[i]) - fast_auc(y[i], b[i]))
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def test_size() -> float:
    try:
        from src.config import get_config
        return get_config().test_size
    except Exception:  # noqa: BLE001
        return 0.20


def split_indices(y, seed):
    """Giong het train_test_split trong stage_2_3_train_oof cua run_pipeline.py."""
    return train_test_split(np.arange(len(y)), test_size=test_size(), stratify=y, random_state=seed)


def md_table(df: pd.DataFrame, floatfmt=4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Baseline learners
# ----------------------------------------------------------------------------------------------
def _design(Z, extra):
    return Z if extra is None else np.column_stack([extra, Z])


def gbm_scores(Ztr, ytr, Zte, seed, extra_tr=None, extra_te=None):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1)
    m.fit(_design(Ztr, extra_tr), ytr)
    return m.predict_proba(_design(Zte, extra_te))[:, 1], m


def lr_scores(Ztr, ytr, Zte, seed, extra_tr=None, extra_te=None):
    """Feature be mat deu >= 0 va lech phai nen log1p truoc; `extra` (diem hidden) giu nguyen."""
    def prep(Z, extra):
        L = np.log1p(Z)
        return L if extra is None else np.column_stack([extra, L])
    sc = StandardScaler().fit(prep(Ztr, extra_tr))
    m = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    m.fit(sc.transform(prep(Ztr, extra_tr)), ytr)
    return m.decision_function(sc.transform(prep(Zte, extra_te))), (sc, m)


def tfidf_scores(q_tr, a_tr, y_tr, q_te, a_te, mode, analyzer, seed):
    def vec():
        if analyzer == "word":
            return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   token_pattern=r"(?u)\b\w+\b", max_features=300000)
        # giu nguyen hoa/thuong de bat duoc van phong
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                               lowercase=False, max_features=300000)
    srcs = {"answer": [("a", a_tr, a_te)], "question": [("q", q_tr, q_te)],
            "both": [("q", q_tr, q_te), ("a", a_tr, a_te)]}[mode]
    btr, bte, names = [], [], []
    for pre, tr, te in srcs:
        v = vec()
        btr.append(v.fit_transform(tr))
        bte.append(v.transform(te))
        names += [f"{pre}:{n}" for n in v.get_feature_names_out()]
    Xtr, Xte = sparse.hstack(btr).tocsr(), sparse.hstack(bte).tocsr()
    clf = LogisticRegressionCV(Cs=[0.1, 0.3, 1, 3, 10], cv=3, scoring="roc_auc", solver="liblinear",
                               max_iter=1000, random_state=seed)
    clf.fit(Xtr, y_tr)  # C chon bang CV tren train, khong dung test
    return clf.decision_function(Xte), clf, names


def top_ngrams(clf, names, k=30) -> pd.DataFrame:
    coef = clf.coef_[0]
    order = np.argsort(coef)
    rows = [("hallucinated", names[i], float(coef[i])) for i in order[::-1][:k]]
    rows += [("faithful", names[i], float(coef[i])) for i in order[:k]]
    return pd.DataFrame(rows, columns=["direction", "ngram", "coef"])


def b1_importance(model, Zte, yte, seed) -> pd.DataFrame:
    r = permutation_importance(model, Zte, yte, scoring="roc_auc", n_repeats=10, random_state=seed)
    df = pd.DataFrame({"feature": ALL_COLS, "perm_auc_drop": r.importances_mean})
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(Zte)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        df["shap_mean_abs"] = np.abs(sv).mean(0)
    except Exception:  # noqa: BLE001  (shap la tuy chon)
        pass
    return df


def baseline_scores(Z, Q, A, tr, te, y, seed, full, want_b2=True):
    """B0/B1/B2 tren split (tr, te). full=False: chi 4 baseline re (dung cho perturbation)."""
    ytr = y[tr]
    Q, A = np.asarray(Q, dtype=object), np.asarray(A, dtype=object)
    preds, extras = {}, {}
    preds["b0_gbm"] = gbm_scores(Z[tr][:, I_B0], ytr, Z[te][:, I_B0], seed)[0]
    preds["b1_gbm"], extras["b1_gbm_model"] = gbm_scores(Z[tr], ytr, Z[te], seed)
    if full:
        preds["b0_lr"] = lr_scores(Z[tr][:, I_B0], ytr, Z[te][:, I_B0], seed)[0]
        preds["b1_lr"], extras["b1_lr"] = lr_scores(Z[tr], ytr, Z[te], seed)
    if want_b2:
        jobs = [("answer", "word"), ("answer", "char")]
        if full:
            jobs += [("question", "word"), ("question", "char"), ("both", "word"), ("both", "char")]
        for mode, an in jobs:
            s, clf, names = tfidf_scores(list(Q[tr]), list(A[tr]), ytr, list(Q[te]), list(A[te]),
                                         mode, an, seed)
            preds[f"b2_{mode}_{an}"] = s
            if full and mode == "answer":
                extras[f"top_{an}"] = top_ngrams(clf, names)
    return preds, extras


# ----------------------------------------------------------------------------------------------
# B3: XLM-R fine-tune tren answer
# ----------------------------------------------------------------------------------------------
def b3_scores(a_tr, y_tr, a_te, seed, args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed_all(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    tok = AutoTokenizer.from_pretrained(args.b3_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.b3_model, num_labels=1).to(dev)
    # Bang embedding cua XLM-R (250k x 768) chiem ~70% tham so: dong bang de vua GPU 6 GB.
    for n, p in model.named_parameters():
        if "word_embeddings" in n:
            p.requires_grad = False

    y_tr = np.asarray(y_tr)
    a_tr = np.asarray(a_tr, dtype=object)
    i_tr, i_va = train_test_split(np.arange(len(a_tr)), test_size=args.b3_val_frac,
                                  stratify=y_tr, random_state=seed)

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=args.b3_maxlen)["input_ids"]
    ids_tr, ids_va, ids_te = enc(a_tr[i_tr]), enc(a_tr[i_va]), enc(a_te)

    def predict(ids):
        model.eval()
        order = np.argsort([len(x) for x in ids])
        res = np.zeros(len(ids))
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            for s in range(0, len(ids), 128):
                b = order[s:s + 128]
                batch = tok.pad({"input_ids": [ids[i] for i in b]}, return_tensors="pt").to(dev)
                res[b] = model(**batch).logits.float().reshape(-1).cpu().numpy()
        return res

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.b3_lr, weight_decay=0.01)
    steps = args.b3_epochs * int(np.ceil(len(ids_tr) / args.b3_bs))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    yt = torch.tensor(y_tr[i_tr], dtype=torch.float32)
    best_val, best_te = -1.0, None
    for ep in range(args.b3_epochs):
        model.train()
        perm = np.random.permutation(len(ids_tr))
        for s in range(0, len(perm), args.b3_bs):
            b = perm[s:s + args.b3_bs]
            batch = tok.pad({"input_ids": [ids_tr[i] for i in b]}, return_tensors="pt").to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logit = model(**batch).logits.float().reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logit, yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step()
        va_auc = roc_auc_score(y_tr[i_va], predict(ids_va))
        print(f"    [b3] epoch {ep + 1}/{args.b3_epochs} val AUROC {va_auc:.4f}", flush=True)
        if va_auc > best_val:
            best_val, best_te = va_auc, predict(ids_te)
    del model, opt
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return best_te, best_val


# ----------------------------------------------------------------------------------------------
# Hidden-state model: tai su dung stage 2/3/4 cua run_pipeline.py
# ----------------------------------------------------------------------------------------------
_RP = None


def _rp():
    global _RP
    if _RP is None:
        import run_pipeline as rp
        _RP = rp
    return _RP


def hs_namespace(dataset_key, lang, model, tag, stack_mode, bs):
    return argparse.Namespace(dataset=dataset_key, lang=lang, model=model, tag=tag,
                              stack_mode=stack_mode, extract_batch_size=bs)


def hs_stack_mode(args):
    return args.stack_mode or ("logit" if args.hs_variant == "improved" else "vector")


def hs_config(seed, variant):
    cfg = _rp().get_config()
    cfg.seed = seed
    if variant == "improved":  # = --improved cua run_pipeline.py
        cfg.use_layer_delta = cfg.scale_layer_weights = cfg.use_answer_ll = True
        cfg.rank_loss_weight = 0.1
    return cfg


def ensure_features(dataset_key, lang, model, args, samples=None):
    """Dam bao results/features/*_{dataset_key}_{lang}_{model}.npy ton tai; tra ve y cua feature."""
    rp = _rp()
    key = f"{dataset_key}_{lang}_{model}"
    if not Path(f"results/features/X_seq_{key}.npy").exists():
        print(f"[hs] trich feature {key} ...", flush=True)
        if samples is not None:
            # stage_1_extract doc dataset khac 'halueval' qua load_triviaqa -> thay bang mau da perturb
            rp.load_triviaqa = lambda lang="en", seed=42, max_samples=10000: samples
        rp.stage_1_extract(hs_namespace(dataset_key, lang, model, "", "logit", args.extract_batch_size),
                           rp.get_config())
    return np.load(f"results/features/y_{key}.npy")


def _hs_cache(ns):
    key = _rp()._key(ns)
    return key, RESULTS / "_cache" / f"hs_{key}.npz"


def hs_is_cached(dataset_key, lang, model, seed, args):
    ns = hs_namespace(dataset_key, lang, model, f"rq2_{args.hs_variant}_s{seed}", hs_stack_mode(args),
                      args.extract_batch_size)
    return _hs_cache(ns)[1].exists() and not args.no_reuse


def run_hs(dataset_key, lang, model, seed, args, expected_y_test):
    """Train (hoac doc cache) mo hinh hidden-state cho 1 seed. Tra ve final (stacker), deep, oof."""
    rp = _rp()
    stack = hs_stack_mode(args)
    ns = hs_namespace(dataset_key, lang, model, f"rq2_{args.hs_variant}_s{seed}", stack,
                      args.extract_batch_size)
    key, cache = _hs_cache(ns)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not args.no_reuse:
        d = np.load(cache)
        out = {k: d[k] for k in d.files}
        print(f"  [hs] doc cache {cache.name}", flush=True)
    else:
        cfg = hs_config(seed, args.hs_variant)
        seed_all(seed)
        needed = ["oof_logits", "test_logits", "X_glob_train", "X_glob_test", "y_train", "y_test"]
        needed += ["X_train_deep", "X_test_deep"] if stack == "vector" else []
        if args.no_reuse or not all(Path(f"results/oof/{n}_{key}.npy").exists() for n in needed):
            rp.stage_2_3_train_oof(ns, cfg)
        cap, orig = {}, rp.plot_publication_results
        # stage_4_ensemble khong tra ve xac suat -> bat lai tham so dau tien cua ham ve hinh
        rp.plot_publication_results = lambda probs, labels, *a, **k: cap.update(
            probs=np.asarray(probs), y=np.asarray(labels))
        try:
            rp.stage_4_ensemble(ns, cfg)
        finally:
            rp.plot_publication_results = orig
        if "probs" not in cap:
            raise RuntimeError(f"stage 4 khong tao ra xac suat cho {key} (thieu feature/OOF?)")
        out = dict(final=cap["probs"],
                   deep=np.load(f"results/oof/test_logits_{key}.npy").mean(0),
                   oof=np.load(f"results/oof/oof_logits_{key}.npy"),
                   y_train=np.load(f"results/oof/y_train_{key}.npy"),
                   y_test=np.load(f"results/oof/y_test_{key}.npy"))
        np.savez(cache, **out)
    if not np.array_equal(out["y_test"], expected_y_test):
        raise RuntimeError("tap test cua pipeline khac split cua RQ2 -> khong so sanh cung split duoc")
    return out


def write_resid_features(Z, tr, te, lang, model, seed, ridge=1.0):
    """Hoi quy (ridge, fit CHI tren train) moi cot feature hidden theo feature be mat roi tru di.
    Ghi ra results/features duoi ten dataset 'halueval-resid<seed>' de stage 2/3 doc nhu binh thuong."""
    src = f"halueval_{lang}_{model}"
    Xs = np.load(f"results/features/X_seq_{src}.npy")
    Xg = np.load(f"results/features/X_glob_{src}.npy")
    y = np.load(f"results/features/y_{src}.npy")
    N, L, F = Xs.shape
    X = np.nan_to_num(np.concatenate([Xs.reshape(N, -1), Xg], axis=1).astype(np.float64))
    lt = np.log1p(Z[:, I_TOK])
    Zf = np.column_stack([Z, np.log1p(Z), lt ** 2, lt ** 3])
    Zs = StandardScaler().fit(Zf[tr]).transform(Zf)
    mu = X[tr].mean(0)
    coef = np.linalg.solve(Zs[tr].T @ Zs[tr] + ridge * np.eye(Zs.shape[1]), Zs[tr].T @ (X[tr] - mu))
    R = X - mu - Zs @ coef
    Xt = X[te]
    ss_res = (R[te] ** 2).sum(0)
    ss_tot = ((Xt - Xt.mean(0)) ** 2).sum(0)
    r2 = float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))
    ds = f"halueval-resid{seed}"
    rk = f"{ds}_{lang}_{model}"
    np.save(f"results/features/X_seq_{rk}.npy", R[:, :L * F].reshape(N, L, F).astype(np.float32))
    np.save(f"results/features/X_glob_{rk}.npy", R[:, L * F:].astype(np.float32))
    np.save(f"results/features/y_{rk}.npy", y)
    return ds, r2


def cleanup_resid(ds, lang, model):
    for p in ("X_seq", "X_glob", "y"):
        f = Path(f"results/features/{p}_{ds}_{lang}_{model}.npy")
        if f.exists():
            f.unlink()


# ----------------------------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------------------------
def length_analysis(y, length, preds, n_bins, seed, n_rep=20) -> pd.DataFrame:
    """AUROC trong tung bin do dai (theo phan vi cua tap test), trung binh co trong so,
    va AUROC tren tap con da can bang nhan trong tung bin (length-matched)."""
    y, length = np.asarray(y), np.asarray(length)
    edges = np.unique(np.quantile(length, np.linspace(0, 1, n_bins + 1)))
    bins = np.digitize(length, edges[1:-1])
    nb = len(edges) - 1
    rows = []
    for m, s in preds.items():
        aucs, ns = [], []
        for b in range(nb):
            idx = bins == b
            ok = idx.sum() > 0 and min(y[idx].sum(), (1 - y[idx]).sum()) >= 10
            auc = fast_auc(y[idx], s[idx]) if ok else np.nan
            rows.append(dict(seed=seed, bin=str(b), lo=edges[b], hi=edges[b + 1], n=int(idx.sum()),
                             pos_rate=float(y[idx].mean()) if idx.sum() else np.nan, model=m, auc=auc))
            aucs.append(auc)
            ns.append(idx.sum())
        w = np.asarray(ns, dtype=float) * ~np.isnan(aucs)
        rows.append(dict(seed=seed, bin="weighted", lo=np.nan, hi=np.nan, n=int(sum(ns)), pos_rate=np.nan,
                         model=m, auc=float(np.nansum(np.nan_to_num(aucs) * w) / w.sum()) if w.sum() else np.nan))
    rng = np.random.default_rng(seed)
    matched = {m: [] for m in preds}
    for _ in range(n_rep):
        keep = []
        for b in range(nb):
            pos, neg = np.where((bins == b) & (y == 1))[0], np.where((bins == b) & (y == 0))[0]
            k = min(len(pos), len(neg))
            if k:
                keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
        keep = np.asarray(keep)
        for m, s in preds.items():
            matched[m].append(fast_auc(y[keep], s[keep]))
    for m in preds:
        rows.append(dict(seed=seed, bin="matched", lo=np.nan, hi=np.nan, n=len(keep), pos_rate=0.5,
                         model=m, auc=float(np.nanmean(matched[m]))))
    return pd.DataFrame(rows)


def complementarity(Z, tr, te, y, hs, seed, boot):
    """Stack diem hidden (OOF logit cua deep net) voi feature be mat va do Delta AUROC (bootstrap theo cap).
    Luu y: tren train la OOF cua 1 net, tren test la trung binh 5 net nen phan phoi hoi khac nhau."""
    ytr, yte = y[tr], y[te]
    s_tr, s_te = hs["oof"][:, None], hs["deep"][:, None]
    rows, scores = [], {}
    for kind, fn in (("lr", lr_scores), ("gbm", gbm_scores)):
        surf = fn(Z[tr], ytr, Z[te], seed)[0]
        both = fn(Z[tr], ytr, Z[te], seed, extra_tr=s_tr, extra_te=s_te)[0]
        scores[f"comp_surf_{kind}"], scores[f"comp_hs+surf_{kind}"] = surf, both
        for label, a, b in (("add_surface_to_hidden", both, hs["deep"]), ("add_hidden_to_surface", both, surf)):
            m, lo, hi = paired_boot(yte, a, b, boot, seed)
            rows.append(dict(seed=seed, learner=kind, comparison=label, auc_a=fast_auc(yte, a),
                             auc_b=fast_auc(yte, b), delta=fast_auc(yte, a) - fast_auc(yte, b),
                             boot_mean=m, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows), scores


# ----------------------------------------------------------------------------------------------
# One (model, language) combination
# ----------------------------------------------------------------------------------------------
def run_combo(args, model, lang):
    from src.data.loader import load_halueval

    out = RESULTS / f"halueval_{lang}_{model}"
    out.mkdir(parents=True, exist_ok=True)
    t_combo = time.perf_counter()
    samples = load_halueval(lang=lang)
    Q = [s["question"] for s in samples]
    A = [s["answer"] for s in samples]
    y = np.array([int(s["is_hallucination"]) for s in samples])
    print(f"\n{'=' * 90}\nRQ2 | halueval | {lang} | {model} | N={len(y)} pos={y.mean():.3f}\n{'=' * 90}",
          flush=True)

    use_hs = "hs" not in args.skip
    if use_hs:
        y_feat = ensure_features("halueval", lang, model, args)
        if len(y_feat) != len(y) or not np.array_equal(y_feat, y):
            print(f"[error] feature da trich ({len(y_feat)}) khong khop mau ({len(y)}) "
                  f"(co batch bi bo qua?). Bo qua to hop nay.", flush=True)
            return False

    count_tokens = make_token_counter(model)
    Z = surface_features(Q, A, lang, count_tokens)
    perts = [] if "perturb" in args.skip else args.perturb
    Zp, Ap = {}, {}
    for p in perts:
        Ap[p] = apply_perturbation(p, A)
        Zp[p] = surface_features(Q, Ap[p], lang, count_tokens)

    # Chan doan ro ri cau hoi giua train/test (split cua pipeline chi stratify theo nhan)
    tr0, te0 = split_indices(y, args.seeds[0])
    q_tr = set(Q[i] for i in tr0)
    both_cls = pd.DataFrame({"q": Q, "y": y}).groupby("q")["y"].nunique()
    overlap = dict(n_unique_questions=len(set(Q)), n_samples=len(Q),
                   test_questions_seen_in_train=float(np.mean([Q[i] in q_tr for i in te0])),
                   questions_with_both_labels=float((both_cls > 1).mean()))
    pd.Series(overlap).to_csv(out / "split_overlap.csv", header=["value"])
    print(f"[split] {overlap}", flush=True)

    rows, bin_rows, comp_rows, imp_rows, pert_rows, single_rows, r2_rows = [], [], [], [], [], [], []
    for seed in args.seeds:
        t0 = time.perf_counter()
        print(f"\n--- seed {seed} ---", flush=True)
        seed_all(seed)
        tr, te = split_indices(y, seed)
        ytr, yte = y[tr], y[te]
        preds, extras = baseline_scores(Z, Q, A, tr, te, y, seed, full=True, want_b2="b2" not in args.skip)

        for j, c in enumerate(ALL_COLS):  # AUROC co dau cua tung feature don le (>0.5: cang lon -> hallucination)
            single_rows.append(dict(seed=seed, feature=c, auc=fast_auc(yte, Z[te][:, j]) if Z[te][:, j].std() > 0 else np.nan))
        for k, df in extras.items():
            if k.startswith("top_"):
                df.to_csv(out / f"top_ngrams_answer_{k[4:]}_seed{seed}.csv", index=False)
        if "importance" not in args.skip:
            d = b1_importance(extras["b1_gbm_model"], Z[te], yte, seed)
            d.insert(0, "seed", seed)
            imp_rows.append(d)
            # he so LR da chuan hoa cua B1
            sc, lr = extras["b1_lr"]
            imp_rows.append(pd.DataFrame({"seed": seed, "feature": ALL_COLS, "lr_coef": lr.coef_[0]}))

        if "b3" not in args.skip:
            preds["b3_xlmr"], val_auc = b3_scores([A[i] for i in tr], ytr, [A[i] for i in te], seed, args)
            print(f"  b3 test AUROC {fast_auc(yte, preds['b3_xlmr']):.4f} (val {val_auc:.4f})", flush=True)

        hs = None
        if use_hs:
            hs = run_hs("halueval", lang, model, seed, args, yte)
            preds["hs_final"], preds["hs_deep"] = hs["final"], hs["deep"]
            if "complement" not in args.skip:
                cdf, cscores = complementarity(Z, tr, te, y, hs, seed, args.boot)
                comp_rows.append(cdf)
                preds.update(cscores)
            if "resid" not in args.skip:
                ds = f"halueval-resid{seed}"
                if not hs_is_cached(ds, lang, model, seed, args):
                    ds, r2 = write_resid_features(Z, tr, te, lang, model, seed)
                    r2_rows.append(dict(seed=seed, mean_r2_surface_to_hidden_test=r2))
                    print(f"  [resid] R^2 trung binh cua feature hidden giai thich boi be mat (test): {r2:.3f}", flush=True)
                rh = run_hs(ds, lang, model, seed, args, yte)
                preds["hs_final_resid"], preds["hs_deep_resid"] = rh["final"], rh["deep"]
                if not args.keep_tmp:
                    cleanup_resid(ds, lang, model)

        for name, s in preds.items():
            rows.append(dict(seed=seed, name=name, auc=fast_auc(yte, s)))

        if "perturb" not in args.skip:
            for p in perts:
                pp, _ = baseline_scores(Zp[p], Q, Ap[p], tr, te, y, seed, full=False, want_b2="b2" not in args.skip)
                if args.perturb_b3 and "b3" not in args.skip:
                    pp["b3_xlmr"] = b3_scores([Ap[p][i] for i in tr], ytr, [Ap[p][i] for i in te], seed, args)[0]
                for name, s in pp.items():
                    pert_rows.append(dict(seed=seed, perturbation=p, name=name, auc=fast_auc(yte, s)))
            for p in (args.hs_perturb if use_hs else []):
                ds = f"halueval-p-{p}"
                ensure_features(ds, lang, model, args, samples=[
                    {**s_, "answer": a_} for s_, a_ in zip(samples, apply_perturbation(p, A))])
                ph = run_hs(ds, lang, model, seed, args, yte)
                pert_rows.append(dict(seed=seed, perturbation=p, name="hs_final", auc=fast_auc(yte, ph["final"])))
                pert_rows.append(dict(seed=seed, perturbation=p, name="hs_deep", auc=fast_auc(yte, ph["deep"])))

        bin_models = [m for m in ["b0_gbm", "b1_gbm", "b2_answer_word", "b2_answer_char", "b3_xlmr",
                                  "hs_final", "hs_deep", "hs_final_resid"] if m in preds]
        bin_rows.append(length_analysis(yte, Z[te][:, I_TOK], {m: preds[m] for m in bin_models},
                                        args.n_bins, seed))
        np.savez(out / f"preds_seed{seed}.npz", te_idx=te, y_test=yte, **preds)
        line = " | ".join(f"{n} {fast_auc(yte, s):.4f}" for n, s in preds.items() if not n.startswith("comp_"))
        print(f"  [seed {seed}] {line}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---------------- aggregate ----------------
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(0, "lang", lang)
    df.insert(0, "dataset", "halueval")
    df.to_csv(out / "per_seed_auroc.csv", index=False)
    pd.concat(bin_rows).to_csv(out / "length_bins.csv", index=False)
    pd.DataFrame(single_rows).to_csv(out / "single_feature_auroc.csv", index=False)
    if comp_rows:
        pd.concat(comp_rows).to_csv(out / "complementarity.csv", index=False)
    if imp_rows:
        pd.concat(imp_rows).to_csv(out / "importance_b1.csv", index=False)
    if pert_rows:
        pd.DataFrame(pert_rows).to_csv(out / "perturbation.csv", index=False)
    if r2_rows:
        pd.DataFrame(r2_rows).to_csv(out / "resid_r2.csv", index=False)
    write_summary(out, df, overlap, model, lang, args)
    print(f"\n[done] {model}/{lang} trong {time.perf_counter() - t_combo:.0f}s -> {out}", flush=True)
    return True


def write_summary(out: Path, df, overlap, model, lang, args):
    pv = df.pivot(index="seed", columns="name", values="auc")
    agg = pd.DataFrame({"mean": pv.mean(), "std": pv.std(ddof=1), "n": pv.count()})
    lines = [f"# RQ2 | halueval | {lang} | {model} | seeds {list(pv.index)}\n"]
    if "hs_final" in pv:
        frac = pv.sub(0.5).div(pv["hs_final"] - 0.5, axis=0)
        agg["explained_vs_hs_final"] = frac.mean()
    agg = agg.reset_index().rename(columns={"name": "model"})
    order = ["b0_lr", "b0_gbm", "b1_lr", "b1_gbm", "b2_answer_word", "b2_answer_char", "b2_question_word",
             "b2_question_char", "b2_both_word", "b2_both_char", "b3_xlmr", "hs_deep", "hs_final",
             "hs_deep_resid", "hs_final_resid", "comp_surf_lr", "comp_hs+surf_lr", "comp_surf_gbm",
             "comp_hs+surf_gbm"]
    agg["o"] = agg["model"].map({n: i for i, n in enumerate(order)}).fillna(99)
    agg = agg.sort_values("o").drop(columns="o")
    agg.to_csv(out / "auroc_summary.csv", index=False)
    lines += ["## AUROC tren tap test (mean +- std qua seed)\n",
              "`explained_vs_hs_final` = (AUROC_baseline - 0.5) / (AUROC_hs_final - 0.5): ti le 'phan vuot chance' "
              "cua mo hinh hidden ma baseline tai hien lai.\n", md_table(agg), ""]

    lb = out / "length_bins.csv"
    if lb.exists():
        d = pd.read_csv(lb)
        d = d[d["bin"].isin(["weighted", "matched"])]
        t = d.groupby(["bin", "model"])["auc"].agg(["mean", "std"]).reset_index()
        lines += ["## AUROC khi kiem soat do dai (bin theo phan vi do dai token cua answer)\n",
                  "`matched`: can bang nhan trong tung bin -> do dai khong con phan biet duoc. "
                  "`weighted`: AUROC trong bin, trung binh co trong so.\n", md_table(t), ""]
    cp = out / "complementarity.csv"
    if cp.exists():
        d = pd.read_csv(cp).groupby(["learner", "comparison"])[["auc_a", "auc_b", "delta", "ci_lo", "ci_hi"]].mean().reset_index()
        lines += ["## Complementarity (delta AUROC; ci = trung binh khoang 95% bootstrap cua tung seed)\n",
                  md_table(d), ""]
    pt = out / "perturbation.csv"
    if pt.exists():
        d = pd.read_csv(pt)
        base = df[df["name"].isin(d["name"].unique())].assign(perturbation="none")[["seed", "perturbation", "name", "auc"]]
        t = pd.concat([base, d]).groupby(["name", "perturbation"])["auc"].mean().unstack("perturbation")
        lines += ["## Perturbation tren answer (train va test deu bi bien doi)\n", md_table(t.reset_index()), ""]
    ip = out / "importance_b1.csv"
    if ip.exists():
        d = pd.read_csv(ip)
        cols = [c for c in ["perm_auc_drop", "shap_mean_abs"] if c in d]
        t = d.groupby("feature")[cols].mean().sort_values(cols[0], ascending=False).head(12).reset_index()
        lines += ["## Feature quan trong nhat cua B1-GBM (permutation / SHAP, trung binh qua seed)\n", md_table(t), ""]
    sf = out / "single_feature_auroc.csv"
    if sf.exists():
        t = pd.read_csv(sf).groupby("feature")["auc"].mean()
        t = t.reindex((t - 0.5).abs().sort_values(ascending=False).index).head(10).reset_index()
        lines += ["## AUROC co dau cua tung feature don le (top 10 lech 0.5 nhat)\n", md_table(t), ""]
    lines += ["## Chan doan split\n", md_table(pd.Series(overlap).rename("value").reset_index()
                                             .rename(columns={"index": "stat"})), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_summary_all():
    parts = [pd.read_csv(p) for p in sorted(RESULTS.glob("halueval_*/per_seed_auroc.csv"))]
    if not parts:
        return
    d = pd.concat(parts)
    s = d.groupby(["dataset", "lang", "model", "name"])["auc"].agg(["mean", "std", "count"]).reset_index()
    s.to_csv(RESULTS / "summary_all.csv", index=False)
    print(f"[summary] {RESULTS / 'summary_all.csv'}", flush=True)


# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="RQ2: surface-feature baselines vs hidden-state detector (HaluEval)",
                                 allow_abbrev=False)
    ap.add_argument("--models", nargs="+", default=["mistral-7b"])
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--datasets", nargs="+", default=["halueval"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--extract_batch_size", type=int, default=4)
    ap.add_argument("--stack_mode", default=None, choices=["logit", "vector"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=SKIPPABLE,
                    help="Bo qua thanh phan: b2 b3 hs resid complement perturb importance")
    ap.add_argument("--hs_variant", default="improved", choices=["improved", "base"])
    ap.add_argument("--no_reuse", action="store_true", help="Train lai hidden-state model, khong dung OOF/cache cu")
    ap.add_argument("--keep_tmp", action="store_true", help="Giu feature residualized trong results/features")
    ap.add_argument("--boot", type=int, default=1000, help="So lan bootstrap cho complementarity")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--perturb", nargs="+", default=PERTURBATIONS, choices=PERTURBATIONS)
    ap.add_argument("--hs_perturb", nargs="*", default=[], choices=PERTURBATIONS,
                    help="Trich lai feature + train lai hidden model tren answer da perturb (rat ton thoi gian)")
    ap.add_argument("--perturb_b3", action="store_true", help="Cho B3 chay ca cac perturbation")
    ap.add_argument("--b3_model", default="xlm-roberta-base")
    ap.add_argument("--b3_epochs", type=int, default=3)
    ap.add_argument("--b3_bs", type=int, default=32)
    ap.add_argument("--b3_lr", type=float, default=3e-5)
    ap.add_argument("--b3_maxlen", type=int, default=96)
    ap.add_argument("--b3_val_frac", type=float, default=0.1)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if "triviaqa" in args.datasets:
        print("[rq2] RQ2 chi chay tren HaluEval -> bo qua triviaqa.", flush=True)
    if "halueval" not in args.datasets:
        print("[rq2] Khong co dataset halueval de chay.", flush=True)
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[rq2] models={args.models} langs={args.langs} seeds={args.seeds} skip={args.skip} "
          f"hs_variant={args.hs_variant} perturb={args.perturb} hs_perturb={args.hs_perturb}", flush=True)
    if args.dry_run:
        print("[rq2] --dry_run: dung o day, khong chay gi.", flush=True)
        return 0
    failed = []
    for model in args.models:
        for lang in args.langs:
            try:
                if not run_combo(args, model, lang):
                    failed.append((model, lang))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                failed.append((model, lang))
                sys.stdout.flush()
    build_summary_all()
    if failed:
        print(f"[rq2] LOI o: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
