"""
Determine the properly-calibrated decision threshold and review-band width
for a reduced physical LAB measurement scheme (default: 2 baseline points +
a single operator's 3 worst-point picks = 5 measurements per part, down
from the original 10), using your real labeled measurement data.

WHY A SEPARATE THRESHOLD PER SCHEME
    Changing which/how many points you average shifts the reference color
    used to compute dE_max, which shifts where the correct pass/fail
    cutoff sits. Reusing a threshold calibrated for the full 10-measurement
    scheme on a reduced scheme is NOT a fair comparison and produces
    misleading error rates - this script derives each scheme's own
    threshold via the same Youden's J method used throughout this project.

WHY A REVIEW BAND
    Parts whose dE_max lands within some margin of the threshold are
    genuinely ambiguous - close enough that measurement noise alone could
    flip the call. Routing just those to a manual boundary-sample check
    (instead of trusting the automatic threshold, or manually checking
    EVERY part) lets you keep the speed benefit of fewer measurements on
    the clear majority of parts while still applying human judgment where
    it's most likely to matter.

Reuses the same point-level cleaning (missing/implausible individual
points get skipped, not treated as a hard failure unless a whole group
ends up empty) and CIEDE2000 math already validated in
thermal_shock_severity_analysis.py, so results are directly comparable to
everything else in this project. Run this script from the same directory
as that file.

Usage:
    python determine_reduced_scheme_threshold.py --csv Denali_Color_Data.csv

    # Try the other operator, or keep more/fewer baseline points:
    python determine_reduced_scheme_threshold.py --csv Denali_Color_Data.csv --operator op2 --baseline_points 3

Requirements:
    pip install pandas numpy scikit-learn --break-system-packages
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from thermal_shock_severity_analysis import (
    BASELINE_COLS, OPERATOR_1_COLS, OPERATOR_2_COLS, coerce_numeric, deltaE2000,
)


def lab_tuple(row, prefix):
    return (row[f"{prefix}_L"], row[f"{prefix}_a"], row[f"{prefix}_b"])


def compute_scheme_dE(df, baseline_cols, worst_cols):
    """
    Computes dE_max per part using only the specified baseline and
    worst-point columns. Same point-level handling as the full analysis:
    a missing/implausible individual point is just skipped, not treated
    as a hard failure unless the whole group (all baseline points, or all
    worst points) ends up empty for that part.
    """
    dEs, valid_idx = [], []
    for idx, row in df.iterrows():
        baseline_pts = [lab_tuple(row, n) for n in baseline_cols
                         if not any(pd.isna(x) for x in lab_tuple(row, n))]
        if not baseline_pts:
            continue
        ref = tuple(np.mean(baseline_pts, axis=0))

        worst_pts = [lab_tuple(row, n) for n in worst_cols
                     if not any(pd.isna(x) for x in lab_tuple(row, n))]
        if not worst_pts:
            continue

        dEs.append(max(deltaE2000(ref, pt) for pt in worst_pts))
        valid_idx.append(idx)

    return pd.Series(dEs, index=valid_idx)


def find_optimal_threshold(dE, y):
    """Youden's J statistic - same method used throughout this project."""
    fpr, tpr, thresholds = roc_curve(y, dE)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def review_band_table(dE, y, threshold, margins):
    pred = (dE >= threshold).astype(int)
    is_error = (pred != y)
    n_errors = int(is_error.sum())

    print(f"\n{'Band half-width':<18}{'% of parts reviewed':<22}{'% of errors caught':<22}{'errors still missed'}")
    print("-" * 80)
    rows = []
    for margin in margins:
        in_band = (dE >= threshold - margin) & (dE <= threshold + margin)
        pct_reviewed = float(in_band.mean())
        errors_caught = int((is_error & in_band).sum())
        pct_caught = errors_caught / n_errors if n_errors else float("nan")
        errors_missed = n_errors - errors_caught
        print(f"+/-{margin:<15.2f}{pct_reviewed:<22.1%}{pct_caught:<22.1%}{errors_missed}")
        rows.append({
            "margin": margin, "pct_of_parts_reviewed": pct_reviewed,
            "pct_of_errors_caught": pct_caught, "errors_still_missed": errors_missed,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to your raw LAB measurement CSV")
    parser.add_argument("--margins", type=float, nargs="+",
                         default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                         help="Review-band half-widths to test, in dE units")
    parser.add_argument("--baseline_points", type=int, default=2,
                         help="How many of the 4 baseline points to keep (first N)")
    parser.add_argument("--operator", choices=["op1", "op2"], default="op1",
                         help="Which single operator's 3 worst-point picks to keep")
    parser.add_argument("--output_csv", default="review_band_tradeoff.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = coerce_numeric(df)
    labels_binary = (df["label"].str.upper() == "FAIL").astype(int)

    baseline_cols = BASELINE_COLS[:args.baseline_points]
    worst_cols = OPERATOR_1_COLS if args.operator == "op1" else OPERATOR_2_COLS

    print(f"Scheme: {len(baseline_cols)} baseline points ({', '.join(baseline_cols)}) "
          f"+ {args.operator}'s 3 worst-point picks "
          f"= {len(baseline_cols) + 3} total measurements per part")

    dE = compute_scheme_dE(df, baseline_cols, worst_cols)
    y = labels_binary.loc[dE.index]
    print(f"{len(dE)} of {len(df)} parts produced a usable dE_max under this scheme")

    # Full 10-measurement scheme, for direct comparison
    dE_full = compute_scheme_dE(df, BASELINE_COLS, OPERATOR_1_COLS + OPERATOR_2_COLS)
    y_full = labels_binary.loc[dE_full.index]
    auc_full = roc_auc_score(y_full, dE_full)
    auc_reduced = roc_auc_score(y, dE)

    print(f"\nAUC - full 10-measurement scheme: {auc_full:.4f}")
    print(f"AUC - this reduced scheme:        {auc_reduced:.4f}")

    threshold, sens, spec = find_optimal_threshold(dE, y)
    print(f"\nOptimal threshold for THIS scheme (Youden's J): {threshold:.3f}")
    print(f"  At this threshold -> sensitivity (fail recall): {sens:.3f}, "
          f"specificity (pass recall): {spec:.3f}")
    print("(This is specific to this exact scheme - re-run this script if you change")
    print(" --baseline_points or --operator, rather than reusing this number elsewhere.)")

    print(f"\n=== Review-band tradeoff ===")
    print("(Parts within +/-margin of the threshold get a human boundary-sample check")
    print(" instead of an automatic call.)")
    band_df = review_band_table(dE, y, threshold, args.margins)

    band_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved the full tradeoff table to {args.output_csv}")
    print("Pick the row whose '% of parts reviewed' and '% of errors caught' you're")
    print("comfortable trading off - there's no single 'correct' band width, it depends")
    print("on how much manual review capacity you actually have.")


if __name__ == "__main__":
    main()
