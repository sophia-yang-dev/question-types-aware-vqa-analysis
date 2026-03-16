#!/usr/bin/env python3
"""
milestone1_full_inference/analyze_results.py

Runs BOTH strict and relaxed evaluation on predictions/all_predictions.jsonl
and saves results to two subdirectories:
  results/strict/    — standard exact-match after basic normalization
  results/relaxed/   — extended normalization (configurable below)

Each subdirectory contains:
  accuracy_5x5_blip.csv / accuracy_5x5_vilt.csv / accuracy_5x5_gap.csv
  accuracy_heatmap_blip.png / accuracy_heatmap_vilt.png / accuracy_heatmap_gap.png
  accuracy_by_group.csv
  summary.txt

Additionally saves to results/:
  per_question_stats.csv   — one row per question, correctness under both modes

Usage:
  python analyze_results.py
  python analyze_results.py --predictions predictions/all_predictions.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
MILESTONE_DIR    = Path(__file__).resolve().parent
PREDICTIONS_FILE = MILESTONE_DIR / "predictions" / "all_predictions.jsonl"
RESULTS_DIR      = MILESTONE_DIR / "results"

# ── GQA 5×5 matrix definition ─────────────────────────────────────────────────
STRUCTURAL_TYPES = ["query", "verify", "logical", "choose", "compare"]
SEMANTIC_TYPES   = ["rel", "attr", "obj", "cat", "global"]

VALID_CELLS = {
    ("query",   "rel"),  ("query",   "attr"), ("query",   "cat"), ("query",   "global"),
    ("verify",  "rel"),  ("verify",  "attr"), ("verify",  "obj"), ("verify",  "global"),
    ("logical", "attr"), ("logical", "obj"),
    ("choose",  "rel"),  ("choose",  "attr"), ("choose",  "cat"), ("choose",  "global"),
    ("compare", "attr"),
}

# ── Capability group definitions ───────────────────────────────────────────────
STRUCTURAL_GROUPS = {
    "S1 Open-ended retrieval (query)":          lambda s, _: s == "query",
    "S2 Binary perception (verify + logical)":  lambda s, _: s in ("verify", "logical"),
    "S3 Constrained choice (choose + compare)": lambda s, _: s in ("choose", "compare"),
}
SEMANTIC_GROUPS = {
    "V1 Relational/spatial (rel)":    lambda _, m: m == "rel",
    "V2 Attribute recognition (attr)": lambda _, m: m == "attr",
    "V3 Object detection (obj)":      lambda _, m: m == "obj",
    "V4 Categorization (cat)":        lambda _, m: m == "cat",
    "V5 Scene understanding (global)": lambda _, m: m == "global",
}

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION — edit this section to change evaluation behaviour
# ══════════════════════════════════════════════════════════════════════════════

_NUM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Prepositions used by the prep-prefix relaxation
_PREPS = frozenset({
    "in", "on", "at", "near", "by", "of", "to", "into", "onto",
    "over", "under", "from", "with", "behind", "above", "below",
})


def normalize_strict(s: str) -> str:
    """
    Standard normalization — identical to what was applied during inference.
    Applied to both prediction and ground truth before comparing.
      1. strip + lowercase
      2. number words → digits
      3. remove leading article (a / an / the)
    """
    s = s.strip().lower()
    if s in _NUM_MAP:
        s = _NUM_MAP[s]
    s = re.sub(r"^(a |an |the )", "", s)
    return s


def _simple_depluralize(word: str) -> str:
    """
    Conservative singular/plural normalizer (no external dependencies).
    Applied word-by-word inside normalize_relaxed.
    Only strips when the result is clearly a valid English stem.
    """
    if len(word) <= 3:
        return word                          # too short to safely strip
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"              # berries → berry
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"              # leaves → leaf
    if word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        return word[:-2]                    # boxes→box, buses→bus, benches→bench
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]                    # zebras→zebra, chairs→chair
    return word


def normalize_relaxed(s: str) -> str:
    """
    Extended normalization for relaxed evaluation.
    Built on top of normalize_strict — modify this function freely.

    Currently handles:
      1. Everything in normalize_strict (applied first)
      2. Prepositional prefix: strips a leading preposition
            "on snow" → "snow", "in park" → "park"
      3. Plural/singular: depluralize each word independently
            "zebras" → "zebra", "train cars" → "train car"

    Ideas for future extension (uncomment or add your own):
      # Superset relaxation — count correct if GT word appears in prediction:
      #   Handled separately in apply_norm_to_df() since it needs both strings.
      # Synonym sets — e.g. {"phone", "cell phone", "mobile"} all equivalent.
      # Abbreviation expansion — e.g. "tv" ↔ "television"
    """
    s = normalize_strict(s)

    # 2. Strip leading preposition  ("on snow" → "snow")
    words = s.split()
    if len(words) >= 2 and words[0] in _PREPS:
        s = " ".join(words[1:])
        words = s.split()

    # 3. Depluralize each word ("zebras" → "zebra")
    s = " ".join(_simple_depluralize(w) for w in words)

    return s


# ── How correctness is computed for each mode ─────────────────────────────────
# Default: normalize both sides and do exact-match.
# You could override this for more aggressive relaxations (e.g. superset match).

def is_correct(pred: str, gt: str, norm_fn) -> bool:
    """
    Return True if pred matches gt under norm_fn.
    Extend this function (or add a separate one) for non-symmetric relaxations
    such as substring / superset matching.
    """
    return norm_fn(str(pred)) == norm_fn(str(gt))


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE — generally no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} predictions from {path.name}")
    return df


def add_correctness_columns(df: pd.DataFrame, norm_fn, suffix: str) -> pd.DataFrame:
    """Add blip_correct_{suffix} and vilt_correct_{suffix} columns using norm_fn."""
    df = df.copy()
    for model, ans_col in [("blip", "blip_answer"), ("vilt", "vilt_answer")]:
        col = f"{model}_correct_{suffix}"
        df[col] = df.apply(
            lambda r: is_correct(r[ans_col], r["gt_answer"], norm_fn)
            if pd.notna(r[ans_col]) else None,
            axis=1,
        )
    return df


def build_matrix(df: pd.DataFrame, correct_col: str) -> pd.DataFrame:
    matrix = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
    matrix[:] = np.nan
    valid = df[correct_col].notna()
    for (struct, sem) in VALID_CELLS:
        mask = (df["structural"] == struct) & (df["semantic"] == sem) & valid
        sub = df.loc[mask, correct_col]
        if len(sub) > 0:
            matrix.loc[struct, sem] = sub.mean()
    return matrix


def build_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(index=STRUCTURAL_TYPES, columns=SEMANTIC_TYPES, dtype=float)
    matrix[:] = np.nan
    for (struct, sem) in VALID_CELLS:
        n = ((df["structural"] == struct) & (df["semantic"] == sem)).sum()
        if n > 0:
            matrix.loc[struct, sem] = n
    return matrix


def print_matrix(matrix: pd.DataFrame, title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")
    print(f"{'':12s}" + "".join(f"{c:>10s}" for c in SEMANTIC_TYPES))
    print("-" * 70)
    for struct in STRUCTURAL_TYPES:
        row = f"{struct:12s}"
        for sem in SEMANTIC_TYPES:
            v = matrix.loc[struct, sem]
            row += f"{'—':>10s}" if pd.isna(v) else f"{v:>10.3f}"
        print(row)
    print("-" * 70)
    # Column (semantic) averages
    row = f"{'SEM AVG':12s}"
    for sem in SEMANTIC_TYPES:
        vals = [matrix.loc[s, sem] for s in STRUCTURAL_TYPES if not pd.isna(matrix.loc[s, sem])]
        row += f"{np.mean(vals):>10.3f}" if vals else f"{'—':>10s}"
    print(row)
    print()
    # Row (structural) averages
    print("STRUCT AVG:")
    for struct in STRUCTURAL_TYPES:
        vals = [matrix.loc[struct, s] for s in SEMANTIC_TYPES if not pd.isna(matrix.loc[struct, s])]
        if vals:
            print(f"  {struct:12s}  {np.mean(vals):.3f}  (over {len(vals)} cells)")


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path,
                 cmap: str = "RdYlGn", vmin=0.0, vmax=1.0,
                 center=None, fmt_fn=None):
    annot = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    for struct in STRUCTURAL_TYPES:
        for sem in SEMANTIC_TYPES:
            v = matrix.loc[struct, sem]
            if not pd.isna(v):
                annot.loc[struct, sem] = fmt_fn(v) if fmt_fn else f"{v:.3f}"

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        matrix.astype(float), annot=annot, fmt="",
        cmap=cmap, vmin=vmin, vmax=vmax, center=center,
        linewidths=0.5, linecolor="white", ax=ax,
        mask=matrix.isna(), annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8},
    )
    for si, struct in enumerate(STRUCTURAL_TYPES):
        for mi, sem in enumerate(SEMANTIC_TYPES):
            if pd.isna(matrix.loc[struct, sem]):
                ax.add_patch(plt.Rectangle((mi, si), 1, 1, fill=True, color="#d0d0d0", lw=0))

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Semantic type", fontsize=11)
    ax.set_ylabel("Structural type", fontsize=11)
    ax.set_xticklabels(SEMANTIC_TYPES, fontsize=10)
    ax.set_yticklabels(STRUCTURAL_TYPES, fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def compute_group_stats(df: pd.DataFrame, blip_col: str, vilt_col: str) -> pd.DataFrame:
    rows = []
    for group_name, fn in {**STRUCTURAL_GROUPS, **SEMANTIC_GROUPS}.items():
        mask = df.apply(lambda r: fn(r["structural"], r["semantic"]), axis=1)
        sub = df[mask]
        if len(sub) == 0:
            continue
        row = {"group": group_name, "n": len(sub)}
        for model, col in [("BLIP", blip_col), ("ViLT", vilt_col)]:
            valid = sub[col].notna()
            row[f"{model}_acc"] = sub.loc[valid, col].mean() if valid.any() else float("nan")
            row[f"{model}_n"]   = valid.sum()
        rows.append(row)
    return pd.DataFrame(rows)


def run_evaluation(df: pd.DataFrame, blip_col: str, vilt_col: str,
                   label: str, out_dir: Path):
    """
    Full evaluation pipeline for one normalization mode.
    label    : "strict" or "relaxed" (used in titles and filenames)
    blip_col : column in df with BLIP correctness booleans
    vilt_col : column in df with ViLT correctness booleans
    out_dir  : directory to write all output files
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*70}")
    print(f"  EVALUATION MODE: {label.upper()}")
    print(f"{'#'*70}")

    has_blip = df[blip_col].notna().any()
    has_vilt = df[vilt_col].notna().any()

    summary_lines = [f"Evaluation mode: {label}", f"Questions processed: {len(df):,}"]

    # ── Accuracy matrices ──────────────────────────────────────────────────────
    blip_matrix = vilt_matrix = None

    if has_blip:
        blip_matrix = build_matrix(df, blip_col)
        print_matrix(blip_matrix, f"BLIP ACCURACY [{label}]")
        blip_matrix.to_csv(out_dir / "accuracy_5x5_blip.csv")
        overall = df[blip_col].dropna().mean()
        summary_lines.append(f"BLIP overall accuracy: {overall:.4f}")

    if has_vilt:
        vilt_matrix = build_matrix(df, vilt_col)
        print_matrix(vilt_matrix, f"ViLT ACCURACY [{label}]")
        vilt_matrix.to_csv(out_dir / "accuracy_5x5_vilt.csv")
        overall = df[vilt_col].dropna().mean()
        summary_lines.append(f"ViLT overall accuracy: {overall:.4f}")

    if has_blip and has_vilt:
        gap_matrix = blip_matrix - vilt_matrix
        print_matrix(gap_matrix, f"GAP (BLIP − ViLT) [{label}]")
        gap_matrix.to_csv(out_dir / "accuracy_5x5_gap.csv")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    print(f"\nSaving heatmaps [{label}]:")
    if has_blip:
        plot_heatmap(blip_matrix,
                     f"BLIP Accuracy ({label}) — GQA Val Balanced",
                     out_dir / "accuracy_heatmap_blip.png")
    if has_vilt:
        plot_heatmap(vilt_matrix,
                     f"ViLT Accuracy ({label}) — GQA Val Balanced",
                     out_dir / "accuracy_heatmap_vilt.png")
    if has_blip and has_vilt:
        plot_heatmap(gap_matrix,
                     f"Accuracy Gap BLIP−ViLT ({label}) — positive = BLIP wins",
                     out_dir / "accuracy_heatmap_gap.png",
                     cmap="RdBu_r", vmin=-0.3, vmax=0.3, center=0.0,
                     fmt_fn=lambda v: f"{v:+.3f}")

    # ── Capability groups ──────────────────────────────────────────────────────
    group_df = compute_group_stats(df, blip_col, vilt_col)
    print(f"\n{'='*70}\n  CAPABILITY GROUPS [{label}]\n{'='*70}")
    print(group_df.to_string(index=False))
    group_df.to_csv(out_dir / "accuracy_by_group.csv", index=False)

    # ── Marginals ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  MARGINALS [{label}]\n{'='*70}")
    for axis_name, type_list in [("structural", STRUCTURAL_TYPES), ("semantic", SEMANTIC_TYPES)]:
        print(f"\nBy {axis_name}:")
        for val in type_list:
            sub = df[df[axis_name] == val]
            if len(sub) == 0:
                continue
            row = f"  {val:12s}  n={len(sub):>7,}"
            if has_blip:
                b = sub[blip_col].dropna().mean()
                row += f"  BLIP={b:.3f}"
            if has_vilt:
                v = sub[vilt_col].dropna().mean()
                row += f"  ViLT={v:.3f}"
            if has_blip and has_vilt:
                row += f"  gap={b - v:+.3f}"
            print(row)

    # ── Summary file ──────────────────────────────────────────────────────────
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nAll [{label}] results saved to: {out_dir.relative_to(MILESTONE_DIR)}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FILE)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw predictions
    df = load_predictions(args.predictions)

    # ── Compute correctness under both norms ───────────────────────────────────
    print("\nComputing strict correctness...")
    df = add_correctness_columns(df, normalize_strict,  suffix="strict")

    print("Computing relaxed correctness...")
    df = add_correctness_columns(df, normalize_relaxed, suffix="relaxed")

    # ── Save unified per-question CSV (both modes in one file) ────────────────
    pq_path = RESULTS_DIR / "per_question_stats.csv"
    df.to_csv(pq_path, index=False)
    print(f"Saved per-question stats ({len(df):,} rows, both modes): {pq_path.name}")

    # ── Run full evaluation pipeline for each mode ─────────────────────────────
    run_evaluation(df,
                   blip_col="blip_correct_strict",
                   vilt_col="vilt_correct_strict",
                   label="strict",
                   out_dir=RESULTS_DIR / "strict")

    run_evaluation(df,
                   blip_col="blip_correct_relaxed",
                   vilt_col="vilt_correct_relaxed",
                   label="relaxed",
                   out_dir=RESULTS_DIR / "relaxed")

    # ── Side-by-side comparison ────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("  STRICT vs RELAXED COMPARISON")
    print(f"{'#'*70}")
    for model, strict_col, relaxed_col in [
        ("BLIP", "blip_correct_strict", "blip_correct_relaxed"),
        ("ViLT", "vilt_correct_strict", "vilt_correct_relaxed"),
    ]:
        s = df[strict_col].dropna().mean()
        r = df[relaxed_col].dropna().mean()
        gained = (df[relaxed_col] & ~df[strict_col]).sum()
        print(f"  {model}:  strict={s:.4f}  relaxed={r:.4f}  (+{r-s:.4f})  "
              f"questions flipped correct: {gained:,}")


if __name__ == "__main__":
    main()
