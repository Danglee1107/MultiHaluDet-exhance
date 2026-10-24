"""RQ1 analysis (scripts/rq/surface.py) on synthetic arrays and files: no model, a few seconds.

Run: uv run pytest scripts/rq
"""
import csv

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.rq.surface import MAX_WORDS, analyse_run, bootstrap_indices, stratified_auc, word_strata


def _confounded(n=2000, seed=0):
    """Hallucinated answers are longer; `length` scores only the length, `signal` also the label."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    words = np.clip(rng.poisson(2 + 6 * y), 1, MAX_WORDS)
    strata = word_strata(["w " * k for k in words])
    length = words + rng.normal(0, 0.1, n)
    signal = length + 3 * y + rng.normal(0, 1, n)
    return y, strata, length, signal


def test_stratified_auc_removes_length():
    y, strata, length, signal = _confounded()
    assert roc_auc_score(y, length) > 0.9  # length alone looks like a detector
    assert abs(stratified_auc(y, length, strata) - 0.5) < 0.1  # until length is held equal
    assert stratified_auc(y, signal, strata) > 0.85


def test_stratified_auc_one_label_strata_drop_out():
    y = np.array([0, 1, 0, 0])
    strata = np.array([0, 0, 1, 1])  # stratum 1 has no hallucinated row
    assert stratified_auc(y, np.array([0.1, 0.9, 5.0, 6.0]), strata) == 1.0
    assert np.isnan(stratified_auc(y, np.zeros(4), np.array([0, 1, 2, 2])))


def test_bootstrap_keeps_cell_counts():
    y, strata, _, _ = _confounded(300)
    for b in bootstrap_indices(y, strata, seed=0, n_boot=3):
        for s in np.unique(strata):
            for c in (0, 1):
                assert ((strata[b] == s) & (y[b] == c)).sum() == ((strata == s) & (y == c)).sum()


def test_word_strata_cap():
    assert list(word_strata(["Paris", " a  b ", "a " * MAX_WORDS, "a " * 50])) == [1, 2, MAX_WORDS, MAX_WORDS]


def test_analyse_run_end_to_end(tmp_path, monkeypatch):
    """Files laid out as run_pipeline.py saves them; the hidden score carries a real signal."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1)
    n, seed = 400, 7
    y = rng.integers(0, 2, n)
    y[:2] = (0, 1)
    samples = [{"question": f"q{i}", "is_hallucination": int(label),
                "answer": "It is the city of Lyon." if label else "Paris"} for i, label in enumerate(y)]
    for s in samples[:60]:
        s["answer"] = "the old port"  # one word count with both labels
    (tmp_path / "results/features").mkdir(parents=True)
    (tmp_path / "results/oof").mkdir()
    np.save("results/features/y_k.npy", y)
    np.save("results/features/X_seq_k.npy", rng.normal(size=(n, 32, 12)))
    np.save("results/features/X_glob_k.npy", rng.normal(size=(n, 34)))
    _, te = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=seed)
    np.save("results/oof/y_test_k_t.npy", y[te])
    np.save("results/oof/test_logits_k_t.npy", y[te] + rng.normal(0, 0.5, (5, len(te))))

    row = analyse_run(samples, "k", "k_t", seed, 0.2, {"tag": "t"}, out_csv="results/rq.csv")
    assert row["reported"] == "network logit" and int(row["n_test"]) == len(te)
    assert float(row["auc_hidden"]) > 0.8
    with open("results/rq.csv") as f:
        assert next(csv.DictReader(f))["verdict"] in ("mostly form", "own signal", "in between")

    np.save("results/oof/y_test_k_t.npy", 1 - y[te])  # a split from another seed is refused
    with pytest.raises(ValueError, match="split"):
        analyse_run(samples, "k", "k_t", seed, 0.2, {}, out_csv="results/rq.csv")
