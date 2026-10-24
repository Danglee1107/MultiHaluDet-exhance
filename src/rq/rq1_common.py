"""Hang so va ham dung chung cho rq1_run.py / rq1_collect.py / rq1_diagnostics.py.

Bo cuc thu muc (moi to hop model x lang co thu muc rieng):
    logs/                         log cua mistral-7b / en  (giu nguyen vi tri cu)
    logs/<model>/<lang>/          log cua cac to hop con lai
    results/rq1/                  CSV cua mistral-7b / en  (giu nguyen vi tri cu) + file tong hop all_*.csv
    results/rq1/<model>/<lang>/   CSV cua cac to hop con lai
Feature va OOF dung chung results/features, results/oof; ten file da chua dataset, lang va model.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "results" / "features"
OOF_DIR = ROOT / "results" / "oof"
LOG_ROOT = ROOT / "logs"
RES_ROOT = ROOT / "results" / "rq1"

DEFAULT_MODEL, DEFAULT_LANG = "mistral-7b", "en"
MODELS = ["mistral-7b", "llama2-7b"]
LANGS = ["en", "fr", "vi", "bn"]
DATASETS = ["halueval", "triviaqa"]
SEEDS = [1, 11, 29, 37, 43]

# Hai dieu kien chi khac nhau 4 cot answer-LL (global 30 -> 34); khong dung --improved.
CONDITIONS = {"base": [], "ll": ["--answer_ll"]}
GLOB_NEEDED = 34


def _cfg_dir(root, model, lang):
    return root if (model, lang) == (DEFAULT_MODEL, DEFAULT_LANG) else root / model / lang


def log_dir(model, lang):
    return _cfg_dir(LOG_ROOT, model, lang)


def out_dir(model, lang):
    return _cfg_dir(RES_ROOT, model, lang)


def feat_key(ds, lang, model):
    return f"{ds}_{lang}_{model}"


def run_key(ds, lang, model, tag):
    return f"{feat_key(ds, lang, model)}_{tag}"


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


def merge_write(path, new, models, langs, datasets):
    """Ghi CSV tong hop all_*.csv theo kieu GOP: bo cac dong cu thuoc (model, lang, dataset) vua xu ly,
    giu nguyen phan con lai, roi them dong moi. Tra ve bang da gop."""
    import pandas as pd
    merged = new
    if path.exists():
        old = pd.read_csv(path)
        if {"model", "lang", "dataset"} <= set(old.columns):
            drop = old["model"].isin(models) & old["lang"].isin(langs) & old["dataset"].isin(datasets)
            old = old[~drop]
            merged = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(merged) or path.exists():
        merged.to_csv(path, index=False)
    return merged


def add_grid_args(p, seeds=True):
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    if seeds:
        p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
