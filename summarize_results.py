"""Tổng hợp AUC base vs v2 theo ngôn ngữ / model / seed từ các log trong logs/multi."""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logdir", default="logs/multi")
ap.add_argument("--out", default="results/summary")
ap.add_argument("--dataset", default=None, help="chỉ tổng hợp dataset này")
args = ap.parse_args()

RES = re.compile(r"Results Summary:\s*\nAUC:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*Acc:\s*([\d.]+)")
DEEP = re.compile(r"Deep-only AUC: OOF logit \(train\) ([\d.]+) \| fold-averaged logit \(test\) ([\d.]+)")
PAIR = re.compile(r"Pair accuracy .*?: ([\d.]+)")
PROBE = re.compile(r"(last|amean): best oof layer (\d+) \(([\d.]+)\)")

def parse(path):
    t = open(path, errors="ignore").read()
    r = RES.search(t)
    if not r:
        return None
    d = {"auc": float(r[1]), "f1": float(r[2]), "acc": float(r[3])}
    m = DEEP.search(t)
    if m:
        d["deep_oof"], d["deep_test"] = float(m[1]), float(m[2])
    m = PAIR.search(t)
    if m:
        d["pair"] = float(m[1])
    d["probe"] = {v: (int(l), float(a)) for v, l, a in PROBE.findall(t)}
    return d

LOGRE = re.compile(r"([^_]+)_([a-z]{2})_(.+)_s(\d+)_(base|v2)\.log$")
LANG_ORDER = ["en", "fr", "vi", "bn"]
runs = {}  # (dataset, lang, model) -> seed -> variant -> dict
for p in glob.glob(os.path.join(args.logdir, "*_s*_*.log")):
    m = LOGRE.match(os.path.basename(p))
    if not m or (args.dataset and m[1] != args.dataset):
        continue
    d = parse(p)
    if d:
        runs.setdefault((m[1], m[2], m[3]), {}).setdefault(int(m[4]), {})[m[5]] = d

f = lambda x: "  -  " if x is None else f"{x:.4f}"
lines, rows = [], []
langkey = lambda k: (k[0], LANG_ORDER.index(k[1]) if k[1] in LANG_ORDER else 99, k[1], k[2])
for key in sorted(runs, key=langkey):
    ds, lang, model = key
    lines += [f"\n=== {ds} | lang={lang} | {model} ===",
              f"{'seed':>5} | {'base AUC':>8} {'v2 AUC':>8} {'Δ':>8} | {'base F1':>7} {'v2 F1':>7} | "
              f"{'base Acc':>8} {'v2 Acc':>7} | {'v2 pair':>7} | probe best (last / amean)"]
    A, B = [], []
    for s in sorted(runs[key]):
        b, v = runs[key][s].get("base"), runs[key][s].get("v2")
        ba, va = (b or {}).get("auc"), (v or {}).get("auc")
        delta = va - ba if ba is not None and va is not None else None
        pr = (v or {}).get("probe", {})
        prs = " / ".join(f"L{pr[k][0]}={pr[k][1]:.3f}" for k in ("last", "amean") if k in pr) or "-"
        lines.append(f"{s:>5} | {f(ba):>8} {f(va):>8} {('%+.4f' % delta) if delta is not None else '  -  ':>8} | "
                     f"{f((b or {}).get('f1')):>7} {f((v or {}).get('f1')):>7} | "
                     f"{f((b or {}).get('acc')):>8} {f((v or {}).get('acc')):>7} | "
                     f"{f((v or {}).get('pair')):>7} | {prs}")
        rows.append([ds, lang, model, s, ba, va, delta, (b or {}).get("f1"), (v or {}).get("f1"),
                     (b or {}).get("acc"), (v or {}).get("acc"), (v or {}).get("pair")])
        if ba is not None and va is not None:
            A.append(ba); B.append(va)
    if A:
        A, B = np.array(A), np.array(B)
        lines.append(f"{'mean':>5} | {A.mean():>8.4f} {B.mean():>8.4f} {(B-A).mean():>+8.4f} |  (n={len(A)} seed, "
                     f"std base {A.std(ddof=1) if len(A)>1 else 0:.4f}, v2 {B.std(ddof=1) if len(B)>1 else 0:.4f}; "
                     f"v2 thắng {int((B>A).sum())}/{len(A)} seed)")

out = "\n".join(lines)
print(out)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out + ".txt", "w").write(out + "\n")
with open(args.out + ".csv", "w") as fh:
    fh.write("dataset,lang,model,seed,base_auc,v2_auc,delta,base_f1,v2_f1,base_acc,v2_acc,v2_pair_acc\n")
    for r in rows:
        fh.write(",".join("" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x)) for x in r) + "\n")
print(f"\nĐã lưu: {args.out}.txt, {args.out}.csv")
