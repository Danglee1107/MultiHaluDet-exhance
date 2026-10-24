"""RQ1 chan doan cho moi to hop model x lang x dataset: answer-LL dung mot minh manh the nao, co trung hidden state khong?

Chay tu root (sau rq1_run.py):
    python src/rq/rq1_diagnostics.py --quiet            # tat ca, chi in mot dong / to hop (chi tiet nam trong log)
    python src/rq/rq1_diagnostics.py --models llama2-7b --langs vi --datasets triviaqa

Dung dung split cua pipeline (train_test_split, stratify, random_state=seed, test_size trong Config). Khong can GPU.
 1. C/D/E voi bo phan loai don gian (LogReg, HistGB), AUC tren tap test:
      C_ll_only (4 cot LL) | D_hidden_only (seq phang + 30 cot global goc) | E_hidden_plus_ll (D + C);  E - D = dong gop cua LL
 2. Tung dac trung LL: AUC don bien (tho: <0.5 la chieu nguoc), Spearman voi logit mang (base), |rho| lon nhat voi 30 cot global goc.
 3. Phan tang theo so token answer (0 / 1 / >=2): so mau, AUC mean-NLL, AUC deep cua base vs ll (can OOF cua rq1_run.py).
 4. meta: ti le nhom n_ans = 0 / 1 / >=2. Nhom 0 = prompt bi cat trong cau hoi; ti le lon (hay gap voi bn, vi) nghia la
    truncation (max_length=256) dang lam hong feature cua ngon ngu do.

Dau ra: results/rq1/[<model>/<lang>/]diag_{conditions,features,strata}_<dataset>.csv, log rq1_diag_<dataset>.log,
        va tong hop results/rq1/all_diag_{conditions,features,strata,meta}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_common import (FEAT_DIR, OOF_DIR, RES_ROOT, ROOT, Tee, add_grid_args, feat_key, log_dir,  # noqa: E402
                        merge_write, out_dir)

sys.path.insert(0, str(ROOT))
from src.config import get_config  # noqa: E402
from src.data.feature_extractor import SEQ_DIM, GLOB_BASE_DIM  # noqa: E402

LL_NAMES = ["ll_mean_nll", "ll_min_logprob", "ll_std_logprob", "ll_mean_entropy"]
N_ANS0_WARN = 0.02


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def make_clf(name, seed):
    if name == "logreg":
        return LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=6,
                                          early_stopping=False, random_state=seed)


def n_ans_group(ll):
    """n_ans khong duoc luu nhung suy ra duoc: n=0 -> ca 4 cot LL = 0; n=1 -> std cua 1 token = 0; con lai n>=2."""
    zero = (ll == 0).all(1)
    one = ~zero & (ll[:, 2] == 0)
    return np.where(zero, "0", np.where(one, "1", ">=2"))


def net_logit(ds, lang, model, cond, seed, n_test):
    p = OOF_DIR / f"test_logits_{ds}_{lang}_{model}_{cond}_s{seed}.npy"
    if not p.exists():
        return None
    z = np.load(p).mean(axis=0)
    return z if len(z) == n_test else None


def run_dataset(model, lang, ds, seeds):
    """Tra ve (cond_df, feat_df, strata_df, meta) hoac None neu chua co feature."""
    cfg = get_config()
    key = feat_key(ds, lang, model)
    try:
        X_seq = np.load(FEAT_DIR / f"X_seq_{key}.npy")[:, :, :SEQ_DIM]
        X_glob = np.load(FEAT_DIR / f"X_glob_{key}.npy")
        y = np.load(FEAT_DIR / f"y_{key}.npy")
    except FileNotFoundError:
        print(f"[{model} {lang} {ds}] chua co feature.")
        return None
    if X_glob.shape[1] < GLOB_BASE_DIM + 4:
        print(f"[{model} {lang} {ds}] feature file chi co {X_glob.shape[1]} cot global, can extract lai (can 34).")
        return None

    N = len(y)
    glob_base, ll = X_glob[:, :GLOB_BASE_DIM], X_glob[:, GLOB_BASE_DIM:GLOB_BASE_DIM + 4]
    hid = np.hstack([X_seq.reshape(N, -1), glob_base])
    mats = {"C_ll_only": ll, "D_hidden_only": hid, "E_hidden_plus_ll": np.hstack([hid, ll])}
    groups = n_ans_group(ll)
    meta = {"model": model, "lang": lang, "dataset": ds, "N": N, "pos_rate": float(y.mean()),
            "frac_n_ans0": float(np.mean(groups == "0")), "frac_n_ans1": float(np.mean(groups == "1")),
            "frac_n_ans_ge2": float(np.mean(groups == ">=2"))}
    print(f"\n##### {model} | {lang} | {ds}: N={N}, hallucination={y.mean():.1%}, "
          f"n_ans: 0={meta['frac_n_ans0']:.1%}  1={meta['frac_n_ans1']:.1%}  >=2={meta['frac_n_ans_ge2']:.1%}")
    if meta["frac_n_ans0"] > N_ANS0_WARN:
        print(f"  [canh bao] {meta['frac_n_ans0']:.1%} mau co n_ans=0 (prompt bi cat truoc answer): "
              f"feature cua to hop nay co the bi hong boi truncation max_length=256.")

    ident = {"model": model, "lang": lang, "dataset": ds}
    cond_rows, feat_rows, strata_rows = [], [], []
    for seed in seeds:
        tr, te = train_test_split(np.arange(N), test_size=cfg.test_size, stratify=y, random_state=seed)
        y_te = y[te]

        for cname, M in mats.items():
            sc = RobustScaler().fit(M[tr])
            Xtr = np.clip(sc.transform(M[tr]), -10, 10)
            Xte = np.clip(sc.transform(M[te]), -10, 10)
            for clf_name in ("logreg", "hgb"):
                clf = make_clf(clf_name, seed).fit(Xtr, y[tr])
                cond_rows.append({**ident, "seed": seed, "condition": cname, "clf": clf_name,
                                  "test_auc": auc(y_te, clf.predict_proba(Xte)[:, 1])})

        z_base = net_logit(ds, lang, model, "base", seed, len(te))
        z_ll = net_logit(ds, lang, model, "ll", seed, len(te))
        ll_te, glob_te = ll[te], glob_base[te]
        valid = groups[te] != "0"
        for j, name in enumerate(LL_NAMES):
            row = {**ident, "seed": seed, "feature": name,
                   "auc_all": auc(y_te, ll_te[:, j]),
                   "auc_n_ans>=1": auc(y_te[valid], ll_te[valid, j])}
            if z_base is not None and valid.sum() > 2:
                row["spearman_vs_net_base"] = spearmanr(ll_te[valid, j], z_base[valid])[0]
            if valid.sum() > 2:
                rho = np.nan_to_num([abs(spearmanr(ll_te[valid, j], glob_te[valid, k])[0])
                                     for k in range(GLOB_BASE_DIM)])
                row["max_abs_spearman_vs_glob30"] = float(rho.max())
                row["argmax_glob30_col"] = int(rho.argmax())
            feat_rows.append(row)

        for g in ("0", "1", ">=2"):
            m = groups[te] == g
            row = {**ident, "seed": seed, "n_ans_group": g, "n_test": int(m.sum()),
                   "pos_rate": float(y_te[m].mean()) if m.any() else np.nan,
                   "auc_mean_nll": auc(y_te[m], ll_te[m, 0]) if m.any() else np.nan}
            if z_base is not None:
                row["deep_auc_base"] = auc(y_te[m], z_base[m]) if m.any() else np.nan
            if z_ll is not None:
                row["deep_auc_ll"] = auc(y_te[m], z_ll[m]) if m.any() else np.nan
            if z_base is not None and z_ll is not None:
                row["delta_deep_auc"] = row["deep_auc_ll"] - row["deep_auc_base"]
            strata_rows.append(row)

    cond_df, feat_df, strata_df = map(pd.DataFrame, (cond_rows, feat_rows, strata_rows))
    d = out_dir(model, lang)
    d.mkdir(parents=True, exist_ok=True)
    for df, name in ((cond_df, "conditions"), (feat_df, "features"), (strata_df, "strata")):
        df.drop(columns=["model", "lang", "dataset"]).to_csv(d / f"diag_{name}_{ds}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    print("\n[1] Test AUC, mean/std qua seed:")
    print(cond_df.groupby(["clf", "condition"])["test_auc"].agg(["mean", "std", "count"]).to_string())
    piv = cond_df.pivot_table(index=["clf", "seed"], columns="condition", values="test_auc")
    piv["E_minus_D"] = piv["E_hidden_plus_ll"] - piv["D_hidden_only"]
    ed = piv.groupby("clf")["E_minus_D"].agg(["mean", "std", lambda s: int((s > 0).sum())])
    ed.columns = ["mean", "std", "n_seed_E>D"]
    print("\n    E - D (dong gop cua 4 cot LL voi bo phan loai don gian):")
    print(ed.to_string())
    print("\n[2] Dac trung LL (mean qua seed):")
    print(feat_df.drop(columns=["model", "lang", "dataset", "seed"]).groupby("feature").mean().to_string())
    print("\n[3] Phan tang theo so token answer (mean qua seed):")
    print(strata_df.drop(columns=["model", "lang", "dataset", "seed"]).groupby("n_ans_group").mean().to_string())
    meta["ll_only_auc_hgb"] = float(cond_df[(cond_df.clf == "hgb") & (cond_df.condition == "C_ll_only")]["test_auc"].mean())
    meta["E_minus_D_hgb"] = float(ed.loc["hgb", "mean"])
    meta["E_minus_D_logreg"] = float(ed.loc["logreg", "mean"])
    return cond_df, feat_df, strata_df, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_grid_args(ap)
    ap.add_argument("--quiet", action="store_true", help="Chi tiet ghi vao log, man hinh chi in mot dong / to hop")
    a = ap.parse_args()

    all_cond, all_feat, all_strata, all_meta = [], [], [], []
    for model in a.models:
        for lang in a.langs:
            for ds in a.datasets:
                ldir = log_dir(model, lang)
                ldir.mkdir(parents=True, exist_ok=True)
                with open(ldir / f"rq1_diag_{ds}.log", "w", encoding="utf-8") as f:
                    old = sys.stdout
                    sys.stdout = f if a.quiet else Tee(old, f)
                    try:
                        res = run_dataset(model, lang, ds, a.seeds)
                    finally:
                        sys.stdout = old
                if res is None:
                    print(f"[bo qua] {model} {lang} {ds}: chua co feature hop le")
                    continue
                cond_df, feat_df, strata_df, meta = res
                all_cond.append(cond_df)
                all_feat.append(feat_df)
                all_strata.append(strata_df)
                all_meta.append(meta)
                if a.quiet:
                    warn = "  [n_ans=0 cao!]" if meta["frac_n_ans0"] > N_ANS0_WARN else ""
                    print(f"{model:11s} {lang} {ds:9s} n_ans0={meta['frac_n_ans0']:5.1%}  LL-only(hgb)={meta['ll_only_auc_hgb']:.3f}  "
                          f"E-D hgb={meta['E_minus_D_hgb']:+.4f} logreg={meta['E_minus_D_logreg']:+.4f}{warn}")

    if not all_meta:
        sys.exit("Khong co to hop nao co feature.")
    RES_ROOT.mkdir(parents=True, exist_ok=True)
    # all_diag_*.csv duoc GOP voi noi dung da co (khong ghi de cac to hop khong nam trong lan chay nay)
    for dfs, name in ((all_cond, "conditions"), (all_feat, "features"), (all_strata, "strata")):
        merge_write(RES_ROOT / f"all_diag_{name}.csv", pd.concat(dfs, ignore_index=True),
                    a.models, a.langs, a.datasets)
    merge_write(RES_ROOT / "all_diag_meta.csv", pd.DataFrame(all_meta), a.models, a.langs, a.datasets)
    print(f"\nCSV tong hop: {RES_ROOT}/all_diag_*.csv")


if __name__ == "__main__":
    main()
