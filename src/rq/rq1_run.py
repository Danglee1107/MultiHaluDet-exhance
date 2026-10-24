"""RQ1 runner: answer-LL bo sung gi so voi hidden state o token cuoi?

Chay tu root:
    python src/rq/rq1_run.py --dry_run          # xem ke hoach, khong chay gi
    python src/rq/rq1_run.py --keep_going       # luoi day du: 2 model x 4 lang x 2 dataset x 5 seed x 2 dieu kien

Mac dinh: models mistral-7b llama2-7b | langs en fr vi bn | datasets halueval triviaqa | seeds 1 11 29 37 43
Chay mot phan:  --models llama2-7b --langs vi --datasets triviaqa --seeds 1 11

Dieu kien (chi khac nhau 4 cot answer-LL):  base (global 30)  |  ll = --answer_ll (global 34).
Du lieu: data/<dataset>_<lang>.json. To hop thieu file (va chua co feature) bi bo qua kem canh bao.
    halueval/en: neu khong co file local thi loader lay tu HuggingFace.
    triviaqa/en: BAT BUOC co data/triviaqa_en.json (hard negatives); thieu thi bo qua, khong dung negative ngau nhien.

Log:   xem rq1_common.py (log_dir). Log tong: logs/rq1_master.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_common import (CONDITIONS, FEAT_DIR, GLOB_NEEDED, LOG_ROOT, OOF_DIR, ROOT, Tee,  # noqa: E402
                        add_grid_args, feat_key, log_dir, run_key)

_DATA_CACHE = {}


def check_data(ds, lang):
    """(ok, thong bao). Kiem tra file du lieu, in thong ke nhan mot lan cho moi (dataset, lang)."""
    if (ds, lang) in _DATA_CACHE:
        return _DATA_CACHE[(ds, lang)]
    path = ROOT / "data" / f"{ds}_{lang}.json"
    rel = path.relative_to(ROOT)
    if not path.exists():
        if ds == "halueval" and lang == "en":
            res = (True, "khong co file local, loader se lay tu HuggingFace (pminervini/HaluEval)")
        elif ds == "triviaqa" and lang == "en":
            res = (False, f"thieu {rel} (loader se roi ve negative = dap an cau hoi ngau nhien)")
        else:
            res = (False, f"thieu {rel}")
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            missing = [k for k in ("question", "answer", "is_hallucination") if k not in data[0]]
            if missing:
                res = (False, f"{rel} thieu key {missing}")
            else:
                labels = np.array([int(d["is_hallucination"]) for d in data])
                msg = f"{rel}: {len(data)} mau, {labels.sum()} hallucination ({labels.mean():.1%})"
                if abs(labels.mean() - 0.5) > 0.1:
                    msg += "  [canh bao: nhan khong can bang]"
                if ds == "triviaqa":
                    msg += "  (loader lay toi da 10000 mau, seed 42)"
                res = (True, msg)
                if ds == "triviaqa":
                    print(f"[data] {msg}")
                    for d in [d for d in data if int(d["is_hallucination"]) == 1][:3]:
                        print(f"         Q: {str(d['question'])[:80]} | neg: {str(d['answer'])[:60]}")
                    _DATA_CACHE[(ds, lang)] = res
                    return res
        except Exception as e:  # JSON hong, rong, ...
            res = (False, f"{rel} khong doc duoc: {type(e).__name__}: {e}")
    print(f"[data] {ds}/{lang}: {res[1]}")
    _DATA_CACHE[(ds, lang)] = res
    return res


def run_logged(cmd, log_path, mode):
    """Chay cmd tai ROOT, vua in ra man hinh vua ghi vao log. Tra ve return code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, mode, encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def feature_files(ds, lang, model):
    key = feat_key(ds, lang, model)
    return FEAT_DIR / f"X_seq_{key}.npy", FEAT_DIR / f"X_glob_{key}.npy"


def features_state(ds, lang, model):
    """'missing' | 'ok' | 'old' (feature file cu, chi 30 cot global)."""
    seq_p, glob_p = feature_files(ds, lang, model)
    if not (seq_p.exists() and glob_p.exists()):
        return "missing"
    return "ok" if np.load(glob_p, mmap_mode="r").shape[1] == GLOB_NEEDED else "old"


def stage_pending(stage, ds, lang, model, tag, force):
    if force:
        return True
    if stage == "train_oof":
        return not (OOF_DIR / f"y_test_{run_key(ds, lang, model, tag)}.npy").exists()
    lp = log_dir(model, lang) / f"rq1_{ds}_{tag}.log"
    return not (lp.exists() and "Results Summary" in lp.read_text(encoding="utf-8", errors="replace"))


def extract(ds, lang, model, a):
    cmd = [sys.executable, "run_pipeline.py", "--dataset", ds, "--lang", lang, "--model", model,
           "--stage", "extract", "--extract_batch_size", str(a.extract_batch_size)]
    return run_logged(cmd, log_dir(model, lang) / f"rq1_extract_{ds}.log", "w")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_grid_args(p)
    p.add_argument("--conditions", nargs="+", default=["base", "ll"], choices=list(CONDITIONS))
    p.add_argument("--stack_mode", default="logit", choices=["logit", "vector"],
                   help="vector chi nen dung cho phu luc; log/tag se co hau to _vec")
    p.add_argument("--stages", nargs="+", default=["extract", "train_oof", "ensemble"],
                   choices=["extract", "train_oof", "ensemble"])
    p.add_argument("--extract_batch_size", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Chay lai train/ensemble ke ca khi da co ket qua")
    p.add_argument("--force_extract", action="store_true", help="Trich xuat lai feature")
    p.add_argument("--keep_going", action="store_true", help="Run loi thi ghi nhan roi chay tiep")
    p.add_argument("--dry_run", action="store_true", help="Chi in ke hoach")
    a = p.parse_args()

    if not a.dry_run:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(sys.__stdout__, open(LOG_ROOT / "rq1_master.log", "a", encoding="utf-8"))
        print(f"\n######## {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    suffix = "_vec" if a.stack_mode == "vector" else ""
    combos, skipped = [], []
    print("\n--- Kiem tra du lieu ---")
    for model in a.models:
        for lang in a.langs:
            for ds in a.datasets:
                ok, msg = check_data(ds, lang)
                state = features_state(ds, lang, model)
                if state == "old" and not a.force_extract:
                    skipped.append((model, lang, ds, "feature file cu (30 cot global): xoa hoac dung --force_extract"))
                elif ok or (state == "ok" and not a.force_extract):
                    combos.append((model, lang, ds, state))
                else:
                    skipped.append((model, lang, ds, msg))

    print("\n--- Ke hoach ---")
    n_train = n_ens = n_ext = 0
    for model, lang, ds, state in combos:
        need_ext = "extract" in a.stages and (state != "ok" or a.force_extract)
        tr = en = 0
        for seed in a.seeds:
            for cond in a.conditions:
                tag = f"{cond + suffix}_s{seed}"
                tr += "train_oof" in a.stages and stage_pending("train_oof", ds, lang, model, tag, a.force)
                en += "ensemble" in a.stages and stage_pending("ensemble", ds, lang, model, tag, a.force)
        total = len(a.seeds) * len(a.conditions)
        print(f"  {model:11s} {lang} {ds:9s} extract: {'CAN CHAY' if need_ext else 'da co':8s} "
              f"train_oof: {tr}/{total}  ensemble: {en}/{total}")
        n_ext += need_ext
        n_train += tr
        n_ens += en
    print(f"Tong: {len(combos)} to hop | {n_ext} extraction | {n_train} train_oof | {n_ens} ensemble")
    for model, lang, ds, why in skipped:
        print(f"  [bo qua] {model} {lang} {ds}: {why}")
    if a.dry_run:
        return

    failures = []
    for model, lang, ds, state in combos:
        if "extract" in a.stages and (state != "ok" or a.force_extract):
            print(f"\n=== extract | {model} {lang} {ds} ===")
            rc = extract(ds, lang, model, a)
            if rc != 0:
                msg = f"[fail] extract {model} {lang} {ds} rc={rc}, xem {log_dir(model, lang)}/rq1_extract_{ds}.log"
                failures.append(msg)
                if a.keep_going:
                    print(msg)
                    continue
                sys.exit(msg)
        for seed in a.seeds:
            for cond in a.conditions:
                tag = f"{cond + suffix}_s{seed}"
                log_path = log_dir(model, lang) / f"rq1_{ds}_{tag}.log"
                for stage in ("train_oof", "ensemble"):
                    if stage not in a.stages or not stage_pending(stage, ds, lang, model, tag, a.force):
                        continue
                    cmd = [sys.executable, "run_pipeline.py", "--dataset", ds, "--lang", lang, "--model", model,
                           "--seed", str(seed), "--stage", stage, "--stack_mode", a.stack_mode, "--tag", tag,
                           *CONDITIONS[cond]]
                    print(f"\n=== {model} {lang} {ds} | {tag} | {stage} ===")
                    t0 = time.perf_counter()
                    rc = run_logged(cmd, log_path, "w" if stage == "train_oof" else "a")
                    print(f"[{stage}] {time.perf_counter() - t0:.0f}s")
                    if rc != 0:
                        msg = f"[fail] {model} {lang} {ds} {tag} {stage} rc={rc}, xem {log_path}"
                        failures.append(msg)
                        if a.keep_going:
                            print(msg)
                            break
                        sys.exit(msg)

    print("\nXong.")
    for msg in failures:
        print(msg)
    for model, lang, ds, why in skipped:
        print(f"[da bo qua] {model} {lang} {ds}: {why}")
    print("Gom ket qua:  python src/rq/rq1_collect.py   |   chan doan: python src/rq/rq1_diagnostics.py --quiet")


if __name__ == "__main__":
    main()
