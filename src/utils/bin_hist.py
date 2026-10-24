"""Histogram of hallucination rate per answer-length bin.

Location : root/src/utils/bin_hist.py
Input    : root/data/halueval_en.json  (JSON list or JSONL)
Output   : root/results/img/*.png

x axis -> answer-length bin, y axis -> % of answers in that bin that are
hallucinated, with the percentage printed on top of each bar.

Usage (from root):
    python src/utils/bin_hist.py                      # 5 quantile bins, word length
    python src/utils/bin_hist.py --n-bins 8
    python src/utils/bin_hist.py --n-bins 10 --mode uniform --unit char
    python src/utils/bin_hist.py --unit token --tokenizer mistralai/Mistral-7B-v0.1
    python src/utils/bin_hist.py --theme orange          # solid color theme
    python src/utils/bin_hist.py --theme YlOrRd          # gradient: color follows the rate
    python src/utils/bin_hist.py --theme "#8E44AD"       # any hex color

From other code:
    from src.utils.bin_hist import plot_length_bin_hist
    plot_length_bin_hist(n_bins=6)

Supported record layouts (auto-detected, or force with --answer-key/--label-key):
  1. Paired  : {"right_answer": ..., "hallucinated_answer": ...}
               -> each record yields 2 samples (label 0 and label 1)
  2. Flat    : {"answer": ..., "label": 0/1 | "yes"/"no" | true/false}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "halueval_en.json"
DEFAULT_OUT_DIR = ROOT / "results" / "img"

# (correct_key, hallucinated_key) candidates for the paired layout
PAIR_KEYS = [
    ("right_answer", "hallucinated_answer"),
    ("correct_answer", "hallucinated_answer"),
    ("right_response", "hallucinated_response"),
    ("right_summary", "hallucinated_summary"),
]
ANSWER_KEYS = ["answer", "response", "output", "text", "summary"]
LABEL_KEYS = ["label", "hallucination", "is_hallucination", "hallucinated", "ground_truth"]

_POS = {"1", "yes", "true", "hallucination", "hallucinated"}
_NEG = {"0", "no", "false", "correct", "right", "factual", "non-hallucination"}


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
# Solid themes: every bar has the same color.
SOLID_THEMES = {
    "blue": "#3B6FB6",
    "orange": "#D9822B",
    "green": "#4C9F70",
    "red": "#C8433F",
    "purple": "#7B5EA7",
    "teal": "#2A9D8F",
    "gray": "#6B7280",
    "black": "#222222",
}
# Gradient themes: any matplotlib colormap name also works (e.g. viridis, magma,
# coolwarm, YlOrRd, Blues...). Bar color follows its hallucination rate (0-100%).
GRADIENT_EXAMPLES = ["viridis", "magma", "plasma", "coolwarm", "YlOrRd", "Blues", "RdYlGn_r"]
DEFAULT_THEME = "blue"


def resolve_colors(theme: str, rates: np.ndarray) -> list:
    """Return one color per bar for the given theme."""
    if theme.startswith("#"):
        return [theme] * len(rates)
    if theme.lower() in SOLID_THEMES:
        return [SOLID_THEMES[theme.lower()]] * len(rates)
    try:
        cmap = matplotlib.colormaps[theme]
    except KeyError:
        raise ValueError(
            f"Unknown theme '{theme}'. Solid: {list(SOLID_THEMES)}; "
            f"gradient (matplotlib colormap), e.g. {GRADIENT_EXAMPLES}; or a hex like '#8E44AD'."
        ) from None
    # keep the very light end of sequential maps readable on a white background
    return [cmap(0.15 + 0.85 * float(r) / 100) for r in rates]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict]:
    """Load a JSON list, a JSON dict wrapping a list, or JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        lists = [v for v in obj.values() if isinstance(v, list)]
        if not lists:
            raise ValueError(f"{path}: JSON is a dict with no list inside.")
        obj = lists[0]
    if not obj or not isinstance(obj[0], dict):
        raise ValueError(f"{path}: expected a non-empty list of objects.")
    return obj


def _to_label(v) -> int:
    if isinstance(v, (bool, int, float)):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in _POS:
        return 1
    if s in _NEG:
        return 0
    raise ValueError(f"Cannot interpret label value: {v!r}")


def extract_samples(
    records: list[dict],
    answer_key: str | None = None,
    label_key: str | None = None,
    pair_keys: tuple[str, str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return (answer_texts, labels) with label 1 = hallucinated."""
    first = records[0]
    texts: list[str] = []
    labels: list[int] = []

    # --- paired layout -------------------------------------------------------
    pair = pair_keys
    if pair is None and answer_key is None:
        pair = next((p for p in PAIR_KEYS if p[0] in first and p[1] in first), None)
    if pair is not None:
        good, bad = pair
        for r in records:
            texts += [str(r[good]), str(r[bad])]
            labels += [0, 1]
        return texts, np.array(labels)

    # --- flat layout ---------------------------------------------------------
    answer_key = answer_key or next((k for k in ANSWER_KEYS if k in first), None)
    label_key = label_key or next((k for k in LABEL_KEYS if k in first), None)
    if answer_key is None or label_key is None:
        raise KeyError(
            "Could not detect the schema. Keys of the first record: "
            f"{list(first.keys())}. Pass --answer-key/--label-key "
            "(flat) or --pair-keys GOOD BAD (paired)."
        )
    for r in records:
        texts.append(str(r[answer_key]))
        labels.append(_to_label(r[label_key]))
    return texts, np.array(labels)


# --------------------------------------------------------------------------- #
# Length + binning
# --------------------------------------------------------------------------- #
def measure_length(texts: list[str], unit: str = "word", tokenizer: str | None = None) -> np.ndarray:
    if unit == "word":
        return np.array([len(t.split()) for t in texts])
    if unit == "char":
        return np.array([len(t) for t in texts])
    if unit == "token":
        if not tokenizer:
            raise ValueError("--unit token needs --tokenizer <hf-name>")
        from transformers import AutoTokenizer  # lazy import

        tok = AutoTokenizer.from_pretrained(tokenizer)
        return np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])
    raise ValueError(f"Unknown unit: {unit}")


def make_edges(lengths: np.ndarray, n_bins: int, mode: str = "quantile") -> np.ndarray:
    """Bin edges e_0..e_k. Bin i = (e_{i-1}, e_i]; the first bin includes e_0."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if mode == "quantile":  # ~equal samples per bin
        q = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(lengths, q, method="higher"))
    elif mode == "uniform":  # equal-width bins
        edges = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if len(edges) - 1 < n_bins:
        print(f"[warn] only {len(edges) - 1} distinct bins possible (ties in length); asked for {n_bins}.")
    return edges


def assign_bins(lengths: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], lengths, side="left")


def bin_labels(edges: np.ndarray) -> list[str]:
    integer = np.allclose(edges, np.round(edges))
    out = []
    for i in range(len(edges) - 1):
        if integer:
            lo = int(edges[i]) if i == 0 else int(edges[i]) + 1
            hi = int(edges[i + 1])
            out.append(f"{lo}" if lo == hi else f"{lo}–{hi}")
        else:
            out.append(f"{edges[i]:.1f}–{edges[i + 1]:.1f}")
    return out


def hallucination_rate_per_bin(lengths: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    b = assign_bins(lengths, edges)
    k = len(edges) - 1
    n = np.array([(b == i).sum() for i in range(k)])
    pos = np.array([labels[b == i].sum() for i in range(k)])
    rate = np.divide(pos, n, out=np.zeros(k), where=n > 0) * 100
    return n, pos, rate


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_length_bin_hist(
    data_path: Path | str = DEFAULT_DATA,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    n_bins: int = 5,
    mode: str = "quantile",
    unit: str = "word",
    tokenizer: str | None = None,
    answer_key: str | None = None,
    label_key: str | None = None,
    pair_keys: tuple[str, str] | None = None,
    out_name: str | None = None,
    show_n: bool = True,
    theme: str = DEFAULT_THEME,
    dpi: int = 200,
) -> Path:
    records = load_records(Path(data_path))
    texts, labels = extract_samples(records, answer_key, label_key, pair_keys)
    lengths = measure_length(texts, unit, tokenizer)

    edges = make_edges(lengths, n_bins, mode)
    n, pos, rate = hallucination_rate_per_bin(lengths, labels, edges)
    names = bin_labels(edges)
    k = len(names)

    print(f"{len(labels)} samples | overall hallucination rate {labels.mean() * 100:.1f}% | unit={unit} mode={mode}")
    print(f"{'bin':>12} {'n':>7} {'halluc':>7} {'rate':>7}")
    for nm, ni, pi, ri in zip(names, n, pos, rate):
        print(f"{nm:>12} {ni:>7d} {pi:>7d} {ri:>6.1f}%")

    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 * k + 3), 4.8))
    ax.bar(range(k), rate, width=0.95, color=resolve_colors(theme, rate), edgecolor="white")
    for i, r in enumerate(rate):
        ax.text(i, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{nm}\n(n={ni})" if show_n else nm for nm, ni in zip(names, n)])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel(f"Answer length bin ({unit}s, {mode} bins)")
    ax.set_ylabel("Hallucination rate (%)")
    # ax.set_title(f"Hallucination rate by answer length — {Path(data_path).stem}")
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = out_name or f"{Path(data_path).stem}_len_hist_{k}bins_{unit}_{mode}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to halueval_en.json")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-name", default=None, help="output file name (default: auto from settings)")
    p.add_argument("--n-bins", type=int, default=5, help="number of length bins")
    p.add_argument("--mode", choices=["quantile", "uniform"], default="quantile",
                   help="quantile = ~equal samples per bin, uniform = equal-width bins")
    p.add_argument("--unit", choices=["word", "char", "token"], default="word")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name, only for --unit token")
    p.add_argument("--answer-key", default=None, help="flat layout: field holding the answer text")
    p.add_argument("--label-key", default=None, help="flat layout: field holding the hallucination label")
    p.add_argument("--pair-keys", nargs=2, metavar=("GOOD", "BAD"), default=None,
                   help="paired layout: field names of correct / hallucinated answer")
    p.add_argument("--theme", default=DEFAULT_THEME,
                   help=f"bar theme: {list(SOLID_THEMES)} | any matplotlib colormap (gradient by rate) | hex like '#8E44AD'")
    p.add_argument("--hide-n", action="store_true", help="do not show n under each bin label")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    plot_length_bin_hist(
        data_path=a.data, out_dir=a.out_dir, n_bins=a.n_bins, mode=a.mode, unit=a.unit,
        tokenizer=a.tokenizer, answer_key=a.answer_key, label_key=a.label_key,
        pair_keys=tuple(a.pair_keys) if a.pair_keys else None,
        out_name=a.out_name, show_n=not a.hide_n, theme=a.theme, dpi=a.dpi,
    )


if __name__ == "__main__":
    main()
