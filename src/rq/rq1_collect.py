"""Gom log RQ1 cua tat ca to hop model x lang thanh CSV.

Chay tu root:
    python src/rq/rq1_collect.py                         # tat ca model, lang, dataset co log
    python src/rq/rq1_collect.py --models llama2-7b --langs vi

Dau ra:
    results/rq1/all_runs.csv, all_summary.csv, all_paired_per_seed.csv, all_paired_summary.csv   (tong hop, co cot model, lang)
    results/rq1/[<model>/<lang>/]  runs.csv, summary.csv, paired_per_seed.csv, paired_summary.csv   (rieng tung to hop)
Moi run duoc kiem tra cau hinh (chi khac nhau 4 cot answer-LL, seed trong log = seed trong ten file).
Log thieu phan train (vd. train bi bo qua vi OOF da co) thi AUC deep-only duoc tinh lai tu file OOF.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_common import OOF_DIR, RES_ROOT, add_grid_args, log_dir, merge_write, out_dir, run_key  # noqa: E402

LOG_RE = re.compile(r"^rq1_(?P<dataset>[a-z0-9]+)_(?P<cond>[a-z_]+?)_s(?P<seed>\d+)\.log$")
HEADER = re.compile(r"seed=(\d+) mixup=(\w+) cutmix=(\w+) layer_delta=(\w+) layer_scale=(\w+) "
                    r"rank_weight=([\d.]+) answer_ll=(\w+) stack=(\w+)")
INPUT = re.compile(r"Input features: seq \((\d+), (\d+)\), global \((\d+),\)")
FOLD = re.compile(r"Fold (\d+)/(\d+): best validation AUC ([\d.]+)")
DEEP = re.compile(r"Deep-only AUC: OOF logit \(train\) ([\d.]+) \| fold-averaged logit \(test\) ([\d.]+)")
RESULT = re.compile(r"^AUC: ([\d.]+), F1: ([\d.]+), Acc: ([\d.]+)", re.M)
STACK4 = re.compile(r"\(stack mode: (\w+)\)")

METRICS = ["deep_test_auc", "deep_oof_auc", "ens_auc", "ens_f1", "ens_acc", "fold_val_auc_mean"]
PAIRED_METRICS = ["deep_test_auc", "ens_auc"]
GROUP = ["model", "lang", "dataset"]


def parse_log(path, dataset, cond, seed):
    text = path.read_text(encoding="utf-8", errors="replace")
    row = {"dataset": dataset, "cond": cond, "seed": seed, "log": path.name}
    m = HEADER.findall(text)
    if m:
        h = m[-1]
        row.update(cfg_seed=int(h[0]), mixup=h[1], cutmix=h[2], layer_delta=h[3], layer_scale=h[4],
                   rank_weight=float(h[5]), answer_ll=h[6], stack=h[7])
    m = INPUT.findall(text)
    if m:
        row["n_layers"], row["seq_dim"], row["glob_dim"] = map(int, m[-1])
    folds = {int(i): float(a) for i, _, a in FOLD.findall(text)}
    if folds:
        row["n_folds"] = len(folds)
        row["fold_val_auc_mean"] = float(np.mean(list(folds.values())))
        row["fold_val_auc_std"] = float(np.std(list(folds.values())))
    m = DEEP.findall(text)
    if m:
        row["deep_oof_auc"], row["deep_test_auc"] = map(float, m[-1])
    m = RESULT.findall(text)
    if m:
        row["ens_auc"], row["ens_f1"], row["ens_acc"] = map(float, m[-1])
    m = STACK4.findall(text)
    if m and "stack" not in row:
        row["stack"] = m[-1]
    row["complete"] = "deep_test_auc" in row and "ens_auc" in row
    return row


def oof_fallback(row):
    """Log thieu phan train_oof: tinh lai AUC deep-only va so cot global tu file OOF."""
    key = run_key(row["dataset"], row["lang"], row["model"], f"{row['cond']}_s{row['seed']}")
    try:
        y_test = np.load(OOF_DIR / f"y_test_{key}.npy")
        test_logits = np.load(OOF_DIR / f"test_logits_{key}.npy")
        y_train = np.load(OOF_DIR / f"y_train_{key}.npy")
        oof_logits = np.load(OOF_DIR / f"oof_logits_{key}.npy")
        glob_dim = np.load(OOF_DIR / f"X_glob_train_{key}.npy", mmap_mode="r").shape[1]
    except FileNotFoundError:
        return
    from sklearn.metrics import roc_auc_score
    row.setdefault("deep_test_auc", float(roc_auc_score(y_test, test_logits.mean(axis=0))))
    row.setdefault("deep_oof_auc", float(roc_auc_score(y_train, oof_logits)))
    row.setdefault("glob_dim", int(glob_dim))
    if "answer_ll" not in row:
        row["answer_ll"] = str(glob_dim == 34)
        row["config_unverified"] = True
    row["from_oof_files"] = True
    row["complete"] = "deep_test_auc" in row and "ens_auc" in row


def config_problems(r):
    """Dieu kien base/ll chi duoc khac nhau o 4 cot answer-LL; bat moi sai lech."""
    if not isinstance(r.get("answer_ll"), str):
        return "khong doc duoc header config"
    want_ll = r["cond"].startswith("ll")
    want_stack = "vector" if r["cond"].endswith("_vec") else "logit"
    probs = []
    if (r["answer_ll"] == "True") != want_ll:
        probs.append(f"answer_ll={r['answer_ll']}")
    if r.get("glob_dim") != (34 if want_ll else 30):
        probs.append(f"glob_dim={r.get('glob_dim')}")
    if isinstance(r.get("stack"), str) and r["stack"] != want_stack:
        probs.append(f"stack={r['stack']}")
    if r.get("config_unverified") is True:  # khong co header: khong kiem tra duoc cac co con lai
        return "; ".join(probs)
    if r["layer_delta"] != "False" or r["layer_scale"] != "False" or r["rank_weight"] != 0.0:
        probs.append("co cai tien khac dang bat (layer_delta/layer_scale/rank_weight)")
    if r.get("cfg_seed") != r["seed"]:
        probs.append(f"seed trong log={r.get('cfg_seed')} khac ten file")
    return "; ".join(probs)


def collect_config(model, lang, datasets):
    ldir = log_dir(model, lang)
    rows = []
    for path in sorted(ldir.glob("rq1_*.log")) if ldir.exists() else []:
        m = LOG_RE.match(path.name)
        if not m or m["dataset"] not in datasets:
            continue
        row = parse_log(path, m["dataset"], m["cond"], int(m["seed"]))
        row.update(model=model, lang=lang)
        if "deep_test_auc" not in row or "glob_dim" not in row:
            oof_fallback(row)
        rows.append(row)
    return rows


def analyze(ok):
    metrics = [m for m in METRICS if m in ok]
    summary = ok.groupby(GROUP + ["cond"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    per_seed, paired = [], []
    for keys, g in ok.groupby(GROUP):
        ident = dict(zip(GROUP, keys))
        for suffix in ("", "_vec"):
            base, ll = "base" + suffix, "ll" + suffix
            sub = g[g["cond"].isin([base, ll])]
            if not {base, ll} <= set(sub["cond"]):
                continue
            for metric in PAIRED_METRICS:
                piv = sub.pivot(index="seed", columns="cond", values=metric).dropna()
                if piv.empty:
                    continue
                delta = piv[ll] - piv[base]
                for seed, d in delta.items():
                    per_seed.append({**ident, "stack": suffix or "_logit", "metric": metric, "seed": seed,
                                     "base": piv.loc[seed, base], "ll": piv.loc[seed, ll], "delta": d})
                row = {**ident, "stack": suffix or "_logit", "metric": metric, "n_seeds": len(delta),
                       "base_mean": piv[base].mean(), "ll_mean": piv[ll].mean(),
                       "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1) if len(delta) > 1 else np.nan,
                       "n_ll_better": int((delta > 0).sum()), "wilcoxon_p": np.nan, "ttest_p": np.nan}
                if len(delta) >= 2:
                    try:
                        row["wilcoxon_p"] = stats.wilcoxon(delta).pvalue
                    except ValueError:
                        pass
                    row["ttest_p"] = stats.ttest_rel(piv[ll], piv[base]).pvalue
                paired.append(row)
    return summary, pd.DataFrame(per_seed), pd.DataFrame(paired)


def save(df, name, model, lang):
    """Ghi ban rieng cua tung to hop model x lang."""
    sub = df[(df["model"] == model) & (df["lang"] == lang)] if len(df) else df
    d = out_dir(model, lang)
    d.mkdir(parents=True, exist_ok=True)
    sub.drop(columns=["model", "lang"], errors="ignore").to_csv(d / name, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_grid_args(ap, seeds=False)
    a = ap.parse_args()

    rows = []
    for model in a.models:
        for lang in a.langs:
            rows += collect_config(model, lang, a.datasets)
    if not rows:
        sys.exit("Khong thay log rq1_<dataset>_<cond>_s<seed>.log nao.")

    runs = pd.DataFrame(rows)
    runs["config_problem"] = [config_problems(r) for r in rows]
    for col in ("config_unverified", "from_oof_files"):
        if col not in runs:
            runs[col] = False
        runs[col] = runs[col].fillna(False).astype(bool)

    for _, r in runs[runs["config_problem"] != ""].iterrows():
        print(f"[canh bao] {r['model']}/{r['lang']}/{r['log']}: {r['config_problem']}")
    for _, r in runs[runs["config_unverified"]].iterrows():
        print(f"[luu y] {r['model']}/{r['lang']}/{r['log']}: khong co header, khong kiem tra duoc layer_delta/layer_scale/rank_weight")
    for _, r in runs[~runs["complete"]].iterrows():
        print(f"[chua xong] {r['model']}/{r['lang']}/{r['log']}")

    RES_ROOT.mkdir(parents=True, exist_ok=True)
    runs_all = merge_write(RES_ROOT / "all_runs.csv", runs, a.models, a.langs, a.datasets)
    ok = runs[runs["complete"] & (runs["config_problem"] == "")]
    if ok.empty:
        sys.exit("Khong co run hop le de tong hop.")

    summary, per_seed, paired = analyze(ok)
    # all_*.csv duoc GOP voi noi dung da co (khong ghi de cac to hop khong nam trong lan chay nay)
    summary = merge_write(RES_ROOT / "all_summary.csv", summary, a.models, a.langs, a.datasets)
    per_seed = merge_write(RES_ROOT / "all_paired_per_seed.csv", per_seed, a.models, a.langs, a.datasets)
    paired = merge_write(RES_ROOT / "all_paired_summary.csv", paired, a.models, a.langs, a.datasets)
    for model in a.models:
        for lang in a.langs:
            for df, name in ((runs_all, "runs.csv"), (summary, "summary.csv"),
                             (per_seed, "paired_per_seed.csv"), (paired, "paired_summary.csv")):
                if len(df) and ((df["model"] == model) & (df["lang"] == lang)).any():
                    save(df, name, model, lang)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200,
                  "display.float_format", "{:.4f}".format)
    print(f"\n{len(ok)}/{len(runs)} run hop le. CSV tong hop: {RES_ROOT}/all_*.csv\n")
    if paired.empty:
        print("Chua co cap base/ll day du de so sanh.")
        return
    lg = paired[paired["stack"] == "_logit"]
    ens = lg[lg["metric"] == "ens_auc"].set_index(GROUP)
    deep = lg[lg["metric"] == "deep_test_auc"].set_index(GROUP)
    table = pd.DataFrame({
        "n_seeds": ens["n_seeds"], "ens_base": ens["base_mean"], "ens_ll": ens["ll_mean"],
        "d_ens": ens["delta_mean"], "d_deep": deep["delta_mean"],
        "ll>base": ens["n_ll_better"].astype(str) + "/" + ens["n_seeds"].astype(str),
        "t_p": ens["ttest_p"]})
    print("Delta = ll - base (ghep cap theo seed, stack logit):")
    print(table.to_string())
    print("\nDelta ensemble trung binh theo dataset (tren cac to hop model x lang):")
    print(table.groupby("dataset")[["d_ens", "d_deep"]].agg(["mean", "min", "max"]).to_string())
    print("\nLuu y: voi n=5 seed, Wilcoxon co p nho nhat la 0.0625; hay nhin huong, so seed ll > base va do lon hieu ung.")


if __name__ == "__main__":
    main()
