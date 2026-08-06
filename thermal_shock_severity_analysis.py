"""
Phase 1: Build an objective thermal-shock severity score from LAB color
measurements and mask data, and validate it against your existing
boundary-sample pass/fail labels.

WHAT THIS DOES
    1. Cleans the data at the POINT level, not the row level: if a single
       baseline or worst-point measurement is missing/non-numeric or
       clearly wrong, only that point is dropped - the rest of the row's
       measurements are still used. A row is only excluded entirely if an
       ENTIRE group becomes unusable (all 4 baseline points bad, or both
       operators' worst-picks bad). This matters most for FAIL rows, which
       are scarce - losing a whole fail example over one bad cell would be
       wasteful.
    2. For each remaining part, computes CIEDE2000 color difference (dE2000)
       between the part's baseline color (avg of whatever valid outside-zone
       readings remain) and each valid operator-picked "worst" point inside
       the thermal shock zone.
    3. Aggregates that into per-part severity features.
    4. Fits a logistic regression against your historical pass/fail labels
       to see how well these physically-grounded features explain the
       existing decisions, and reports the best single dE threshold as a
       simple, auditable decision rule.

EXPECTED INPUT CSV FORMAT (adjust COLUMN CONFIG below to match yours):
    One row per part, with columns:
        part_id (or image_file_number), label   (label = "pass" or "fail")
        base1_L, base1_a, base1_b           (4 outside-zone baseline points)
        base2_L, base2_a, base2_b
        base3_L, base3_a, base3_b
        base4_L, base4_a, base4_b
        op1_pt1_L, op1_pt1_a, op1_pt1_b     (operator 1's 3 worst-area picks)
        op1_pt2_L, op1_pt2_a, op1_pt2_b
        op1_pt3_L, op1_pt3_a, op1_pt3_b
        op2_pt1_L, op2_pt1_a, op2_pt1_b     (operator 2's 3 worst-area picks)
        op2_pt2_L, op2_pt2_a, op2_pt2_b
        op2_pt3_L, op2_pt3_a, op2_pt3_b
        mask_pixels                         (optional - thermal shock mask pixel count)
        total_pixels                        (optional - total visible part area in pixels)

Requirements:
    pip install pandas numpy scikit-learn --break-system-packages
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict


BASELINE_COLS = ["base1", "base2", "base3", "base4"]
OPERATOR_1_COLS = ["op1_pt1", "op1_pt2", "op1_pt3"]
OPERATOR_2_COLS = ["op2_pt1", "op2_pt2", "op2_pt3"]
WORST_COLS = OPERATOR_1_COLS + OPERATOR_2_COLS

# Broad physical plausibility bounds - catches obvious typos (like a missing
# decimal point turning 7.04 into 70.4) regardless of which group a point
# belongs to. These are intentionally loose; they're a sanity net, not a
# precision filter.
L_BOUNDS = (0.0, 100.0)
AB_BOUNDS = (-50.0, 50.0)


def deltaE76(lab1, lab2):
    """Plain Euclidean distance in LAB space - the formula you've been using
    historically. No perceptual reweighting at all, unlike deltaE2000 above."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def deltaE2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    CIEDE2000 color difference between two LAB colors.
    lab1, lab2: (L, a, b) tuples.
    This is the standard formula (Sharma et al. 2005); more perceptually
    accurate than plain Euclidean distance (CIE76), especially for the
    kind of reddish/brownish shifts thermal discoloration tends to cause.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt((C_avg**7) / (C_avg**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif (h2p - h1p) > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)

    L_avgp = (L1 + L2) / 2.0
    C_avgp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        h_avgp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_avgp = (h1p + h2p) / 2.0
    elif (h1p + h2p) < 360:
        h_avgp = (h1p + h2p + 360) / 2.0
    else:
        h_avgp = (h1p + h2p - 360) / 2.0

    T = (1 - 0.17 * math.cos(math.radians(h_avgp - 30))
         + 0.24 * math.cos(math.radians(2 * h_avgp))
         + 0.32 * math.cos(math.radians(3 * h_avgp + 6))
         - 0.20 * math.cos(math.radians(4 * h_avgp - 63)))

    d_ro = 30 * math.exp(-(((h_avgp - 275) / 25) ** 2))
    RC = 2 * math.sqrt((C_avgp**7) / (C_avgp**7 + 25**7))
    SL = 1 + (0.015 * (L_avgp - 50) ** 2) / math.sqrt(20 + (L_avgp - 50) ** 2)
    SC = 1 + 0.045 * C_avgp
    SH = 1 + 0.015 * C_avgp * T
    RT = -math.sin(math.radians(2 * d_ro)) * RC

    dE = math.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return dE


def lab_tuple(row, prefix):
    return (row[f"{prefix}_L"], row[f"{prefix}_a"], row[f"{prefix}_b"])


def coerce_numeric(df):
    """Convert all LAB columns to numeric; non-numeric entries (like a
    stray letter) become NaN rather than crashing downstream math."""
    df = df.copy()
    all_lab_cols = []
    for prefix in BASELINE_COLS + WORST_COLS:
        for ch in ("L", "a", "b"):
            all_lab_cols.append(f"{prefix}_{ch}")
    for col in all_lab_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def point_is_plausible(L, a, b):
    if pd.isna(L) or pd.isna(a) or pd.isna(b):
        return False
    if not (L_BOUNDS[0] <= L <= L_BOUNDS[1]):
        return False
    if not (AB_BOUNDS[0] <= a <= AB_BOUNDS[1]):
        return False
    if not (AB_BOUNDS[0] <= b <= AB_BOUNDS[1]):
        return False
    return True


def filter_baseline_group(row, l_thresh=5.0, ab_thresh=3.0):
    """
    Returns (reference_lab, dropped_notes, usable) for the 4 baseline points.

    Drops individual points that are missing/implausible, or that disagree
    with the group's median beyond threshold (a strong signal of a typo,
    since these 4 points are supposed to represent a fairly uniform region
    on the same part). The reference color is the mean of whatever points
    survive - so a row with 1 bad baseline point still gets a usable
    reference from the other 3, instead of being thrown out entirely.
    """
    candidates = []
    dropped = []
    for name in BASELINE_COLS:
        L, a, b = lab_tuple(row, name)
        if point_is_plausible(L, a, b):
            candidates.append((name, L, a, b))
        else:
            dropped.append(f"{name} missing/implausible")

    if not candidates:
        return None, dropped, False

    arr = np.array([[L, a, b] for _, L, a, b in candidates])
    median = np.median(arr, axis=0)

    kept = []
    for name, L, a, b in candidates:
        if (abs(L - median[0]) > l_thresh
                or abs(a - median[1]) > ab_thresh
                or abs(b - median[2]) > ab_thresh):
            dropped.append(f"{name} disagrees with other baseline points")
        else:
            kept.append((L, a, b))

    if not kept:
        # Median itself may have been skewed by a majority of bad points -
        # fall back to using all numeric/plausible candidates rather than
        # losing the row entirely.
        kept = [(L, a, b) for _, L, a, b in candidates]

    ref = tuple(np.mean(np.array(kept), axis=0))
    return ref, dropped, True


def filter_operator_group(row, op_cols):
    """
    Returns (valid_points, dropped_notes) for one operator's 3 worst-area
    picks, where valid_points is a list of (name, (L,a,b)) tuples.
    Deliberately no spread-based filtering here - worst points are allowed
    to differ a lot from each other, since real severity varies across the
    defect. Only drops points that are missing or physically implausible
    on an absolute basis (see point_is_plausible) - a separate check
    against the baseline (see filter_by_severity_ceiling) catches points
    that are individually in-bounds but produce an implausible dE.
    """
    valid = []
    dropped = []
    for name in op_cols:
        L, a, b = lab_tuple(row, name)
        if point_is_plausible(L, a, b):
            valid.append((name, (L, a, b)))
        else:
            dropped.append(f"{name} missing/implausible")
    return valid, dropped


def filter_by_severity_ceiling(ref, named_points, ceiling):
    """
    Drops any point whose CIEDE2000 distance from the reference color
    exceeds `ceiling`. A worst-point measurement can be individually
    in-bounds (a valid L/a/b on its own) and still be wrong - e.g. a
    probe placed off the actual defect, or a transposed digit that
    happens to still land within plausible L/a/b ranges. Real thermal
    shock in this kind of dataset tops out well below most sanity
    ceilings you'd set here; a dE this large is far more likely to be a
    measurement error than a genuinely severe part.
    """
    kept, dropped = [], []
    for name, lab in named_points:
        dE = deltaE2000(ref, lab)
        if dE > ceiling:
            dropped.append(f"{name} dE={dE:.1f} exceeds sanity ceiling ({ceiling})")
        else:
            kept.append((name, lab))
    return kept, dropped


def compute_features(df, standard_lab=None, l_thresh=5.0, ab_thresh=3.0, severity_ceiling=20.0):
    """
    Turn raw LAB columns into per-part severity features, using point-level
    cleaning (see filter_baseline_group / filter_operator_group above).

    Returns (features_df, excluded_rows_df). A row lands in excluded_rows_df
    only when a whole group is unusable - all 4 baseline points bad, or
    both operators' worst-picks entirely bad - since there's no way to
    compute a severity number in that case.
    """
    records = []
    excluded = []

    for idx, row in df.iterrows():
        identifier = row.get("part_id", row.get("image_file_number", idx))
        ref, base_dropped, base_ok = filter_baseline_group(row, l_thresh, ab_thresh)

        op1_valid_named, op1_dropped = filter_operator_group(row, OPERATOR_1_COLS)
        op2_valid_named, op2_dropped = filter_operator_group(row, OPERATOR_2_COLS)

        dropped_notes = (base_dropped
                          + [f"op1 {d}" for d in op1_dropped]
                          + [f"op2 {d}" for d in op2_dropped])

        if not base_ok:
            excluded.append({"identifier": identifier,
                              "reason": "all 4 baseline points missing/implausible"})
            continue

        # Now that we have a reference color, catch points that are
        # individually plausible but produce an implausibly large color
        # jump from this part's own baseline (see filter_by_severity_ceiling).
        op1_valid_named, op1_ceiling_dropped = filter_by_severity_ceiling(ref, op1_valid_named, severity_ceiling)
        op2_valid_named, op2_ceiling_dropped = filter_by_severity_ceiling(ref, op2_valid_named, severity_ceiling)
        dropped_notes += [f"op1 {d}" for d in op1_ceiling_dropped]
        dropped_notes += [f"op2 {d}" for d in op2_ceiling_dropped]

        op1_valid = [lab for _, lab in op1_valid_named]
        op2_valid = [lab for _, lab in op2_valid_named]

        if not op1_valid and not op2_valid:
            excluded.append({"identifier": identifier,
                              "reason": "no usable worst-point measurements from either operator"})
            continue

        worst_labs = op1_valid + op2_valid
        worst_dEs = [deltaE2000(ref, lab) for lab in worst_labs]
        worst_dEs_76 = [deltaE76(ref, lab) for lab in worst_labs]
        op1_dEs = [deltaE2000(ref, lab) for lab in op1_valid]
        op2_dEs = [deltaE2000(ref, lab) for lab in op2_valid]

        # Operator agreement only makes sense if BOTH operators contributed
        # at least one valid point this row.
        op_agreement_gap = abs(max(op1_dEs) - max(op2_dEs)) if (op1_dEs and op2_dEs) else np.nan

        extent = None
        if "mask_pixels" in df.columns and pd.notna(row.get("mask_pixels")):
            if "total_pixels" in df.columns and pd.notna(row.get("total_pixels")):
                extent = row["mask_pixels"] / row["total_pixels"]
            else:
                extent = row["mask_pixels"]

        record = {
            "part_id": identifier,
            "label": row["label"],
            "dE_max": max(worst_dEs),
            "dE_max_cie76": max(worst_dEs_76),
            "dE_mean": float(np.mean(worst_dEs)),
            "op_agreement_gap": op_agreement_gap,
            "n_dropped_points": len(dropped_notes),
            "dropped_points": "; ".join(dropped_notes) if dropped_notes else "",
        }

        # Carry through any identifying/grouping columns your CSV has, so
        # they're available later for breaking results down by truck, date,
        # part size, etc. without needing to re-join back to the raw data.
        for passthrough_col in ("truck_number", "date", "well_number", "well_size"):
            if passthrough_col in df.columns:
                record[passthrough_col] = row.get(passthrough_col)

        if extent is not None:
            record["extent"] = extent
            record["dE_x_extent"] = max(worst_dEs) * extent

        if standard_lab is not None:
            worst_dEs_vs_standard = [deltaE2000(lab, standard_lab) for lab in worst_labs]
            record["dE_max_vs_standard"] = max(worst_dEs_vs_standard)
            record["dE_mean_vs_standard"] = float(np.mean(worst_dEs_vs_standard))
            record["dE_base_vs_standard"] = deltaE2000(ref, standard_lab)

        records.append(record)

    return pd.DataFrame(records), pd.DataFrame(excluded)


def find_best_threshold(dE_values, labels_binary):
    """Sweep severity thresholds and report the one that best separates pass/fail."""
    fpr, tpr, thresholds = roc_curve(labels_binary, dE_values)
    j_scores = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def analyze_group_confounding(features, group_col, feature_cols, min_group_size=5):
    """
    Tests whether a grouping variable (e.g. truck_number) explains a
    feature's relationship with the fail label, or whether the relationship
    holds even after accounting for group-level differences.

    Prints:
      1. A per-group summary (fail rate + mean of each feature), so you can
         see directly which trucks run worse.
      2. BETWEEN-group correlation: using group-level averages, does a
         truck's average feature value predict its fail rate? A strong
         relationship here means the pattern operates at the truck level
         (e.g. "worse trucks are both further from standard AND fail more").
      3. WITHIN-group AUC: after subtracting each group's own mean from
         every part's feature value (so we're only looking at how a part
         compares to its OWN truck's typical part, not to the whole
         dataset), does the feature still predict fail? If this collapses
         toward 0.5 while the raw/between-group numbers are strong, the
         original relationship was being driven by which truck a part came
         from, not anything distinguishing individual parts within a truck.
    """
    if group_col not in features.columns:
        print(f"No '{group_col}' column found - skipping group analysis.")
        return

    df = features.copy()
    df["_fail"] = (df["label"].str.lower() == "fail").astype(int)

    group_sizes = df.groupby(group_col).size()
    valid_groups = group_sizes[group_sizes >= min_group_size].index
    df_valid = df[df[group_col].isin(valid_groups)].copy()
    n_excluded_groups = group_sizes.size - len(valid_groups)

    print(f"\n=== Group breakdown by '{group_col}' ===")
    print(f"{len(valid_groups)} group(s) with >= {min_group_size} parts "
          f"({n_excluded_groups} smaller group(s) excluded from this analysis)")

    summary = df_valid.groupby(group_col).agg(
        n_parts=("_fail", "size"),
        fail_rate=("_fail", "mean"),
        **{f"mean_{c}": (c, "mean") for c in feature_cols}
    ).sort_values("fail_rate", ascending=False)
    print(summary.to_string())

    for col in feature_cols:
        print(f"\n--- {col} vs '{group_col}' ---")

        # Between-group: does a group's average feature value predict its fail rate?
        group_means = df_valid.groupby(group_col)[col].mean()
        group_fail_rates = df_valid.groupby(group_col)["_fail"].mean()
        between_corr = np.corrcoef(group_means, group_fail_rates)[0, 1]
        print(f"Between-group correlation (group avg {col} vs group fail rate): {between_corr:.3f}")

        # Within-group: subtract each part's group mean, then check if what's
        # LEFT still predicts fail. This isolates the part-level signal from
        # the group-level signal.
        df_valid[f"_{col}_within"] = df_valid[col] - df_valid.groupby(group_col)[col].transform("mean")
        try:
            within_auc = roc_auc_score(df_valid["_fail"], df_valid[f"_{col}_within"])
        except ValueError:
            within_auc = float("nan")
        raw_auc = roc_auc_score(df_valid["_fail"], df_valid[col])
        print(f"Raw AUC (ungrouped): {raw_auc:.4f}   |   Within-group AUC: {within_auc:.4f}")

        if abs(within_auc - 0.5) < abs(raw_auc - 0.5) * 0.5:
            print(f"  -> Within-group AUC collapsed toward 0.5: this relationship looks like it's")
            print(f"     mostly a {group_col}-level effect, not a part-level one.")
        else:
            print(f"  -> Within-group AUC held up: the relationship exists even comparing parts")
            print(f"     to their own {group_col}'s typical part, not just driven by group averages.")


def crosstab_diagnostic(features, col_a, col_b):
    """
    Checks whether two grouping variables (e.g. truck_number and well_size)
    are entangled with each other - i.e. do certain trucks disproportionately
    produce certain well sizes? If so, an apparent "well_size effect" could
    really just be the truck effect showing up again under a different name.
    """
    if col_a not in features.columns or col_b not in features.columns:
        return

    print(f"\n=== Cross-tab: {col_a} x {col_b} (part counts) ===")
    counts = pd.crosstab(features[col_a], features[col_b])
    print(counts.to_string())

    fail = (features["label"].str.lower() == "fail").astype(int)
    print(f"\n=== Cross-tab: {col_a} x {col_b} (fail rate, blank = no parts of that combo) ===")
    fail_rate = pd.crosstab(features[col_a], features[col_b], values=fail, aggfunc="mean")
    print(fail_rate.round(3).to_string())

    print(f"\nIf each {col_a} concentrates heavily in one {col_b} (mostly one non-zero column")
    print(f"per row above), the two variables are confounded with each other in this dataset -")
    print(f"any effect attributed to one could really belong to the other.")


def within_truck_size_comparison(features, severity_col="dE_max",
                                  truck_col="truck_number", size_col="well_size",
                                  min_n=5):
    """
    Trucks are loaded by SERIES (the first two digits of well_size), so a
    single truck's shipment mixes multiple well sizes within that series
    (e.g. one truck carries both 6024 and 6038). That gives a cleaner test
    than the pooled group comparison: for trucks that shipped more than one
    well_size, compare fail rate and severity BETWEEN those sizes WITHIN
    THE SAME TRUCK. If a size difference shows up even here, it's evidence
    of a real size effect - not just "this truck happens to be bad,"
    since truck is held constant in each comparison.
    """
    if truck_col not in features.columns or size_col not in features.columns:
        print(f"Need both '{truck_col}' and '{size_col}' columns for this comparison.")
        return

    df = features.copy()
    df["_fail"] = (df["label"].str.lower() == "fail").astype(int)

    print(f"\n=== Within-truck comparison: does {size_col} matter even holding {truck_col} fixed? ===")
    rows = []
    for truck, truck_df in df.groupby(truck_col):
        sizes_here = truck_df[size_col].unique()
        if len(sizes_here) < 2:
            continue  # this truck only shipped one size - no within-truck comparison possible

        size_stats = truck_df.groupby(size_col).agg(
            n=("_fail", "size"),
            fail_rate=("_fail", "mean"),
            mean_severity=(severity_col, "mean"),
        )
        size_stats = size_stats[size_stats["n"] >= min_n]
        if len(size_stats) < 2:
            continue  # not enough parts of each size in this truck to compare

        for size_val, row in size_stats.iterrows():
            rows.append({
                truck_col: truck, size_col: size_val,
                "n": int(row["n"]), "fail_rate": row["fail_rate"],
                "mean_severity": row["mean_severity"],
            })

    if not rows:
        print(f"No trucks shipped multiple well sizes with >= {min_n} parts each - "
              f"can't run this comparison with the current data.")
        return

    comparison_df = pd.DataFrame(rows).sort_values([truck_col, size_col])
    print(comparison_df.to_string(index=False))

    # Summarize: for trucks with exactly 2 comparable sizes, how big and
    # consistent is the within-truck fail-rate gap between them?
    gaps = []
    for truck, g in comparison_df.groupby(truck_col):
        if len(g) == 2:
            gaps.append(g["fail_rate"].max() - g["fail_rate"].min())
    if gaps:
        print(f"\nAcross {len(gaps)} truck(s) that shipped 2 comparable sizes, the within-truck")
        print(f"fail-rate gap between sizes averaged {np.mean(gaps):.1%} "
              f"(range {min(gaps):.1%} to {max(gaps):.1%}).")
        print("A consistently large gap here means size matters independent of truck.")
        print("A small/inconsistent gap means the earlier well_size pattern was likely truck-driven.")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to your measurement CSV")
    parser.add_argument("--standard_L", type=float, default=None)
    parser.add_argument("--standard_a", type=float, default=None)
    parser.add_argument("--standard_b", type=float, default=None)
    parser.add_argument("--l_range_thresh", type=float, default=5.0,
                         help="Max allowed deviation of a baseline point's L from the group median")
    parser.add_argument("--ab_range_thresh", type=float, default=3.0,
                         help="Max allowed deviation of a baseline point's a/b from the group median")
    parser.add_argument("--severity_ceiling", type=float, default=20.0,
                         help="Max plausible dE for a single worst-point measurement; points above "
                              "this are dropped as likely measurement errors, not real severity")
    parser.add_argument("--excluded_rows_csv", default="excluded_rows_report.csv",
                         help="Where to save rows that had to be dropped entirely")
    parser.add_argument("--group_col", default="truck_number",
                         help="Column to check for confounding (e.g. truck_number, date, well_size)")
    parser.add_argument("--min_group_size", type=int, default=5,
                         help="Minimum parts per group to include it in the group analysis")
    parser.add_argument("--stratify_by", default=None,
                         help="Column to compute SEPARATE severity thresholds per group "
                              "(e.g. well_size), instead of one pooled threshold")
    parser.add_argument("--features_csv_out", default=None,
                         help="Optional: save the computed per-part features (including dE_max) "
                              "to this CSV path - needed as input for pairing images to severity "
                              "scores for training an image-based regression model")
    args = parser.parse_args()

    standard_lab = None
    if args.standard_L is not None and args.standard_a is not None and args.standard_b is not None:
        standard_lab = (args.standard_L, args.standard_a, args.standard_b)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    df = coerce_numeric(df)
    features, excluded_df = compute_features(df, standard_lab, args.l_range_thresh,
                                              args.ab_range_thresh, args.severity_ceiling)

    if args.features_csv_out:
        export_cols = [c for c in features.columns if c != "dropped_points"]
        features[export_cols].to_csv(args.features_csv_out, index=False)
        print(f"\nSaved computed features (including dE_max) to {args.features_csv_out}")

    print(f"\n{len(excluded_df)} row(s) excluded entirely (whole group unusable); "
          f"{len(features)} rows retained for analysis")
    if len(excluded_df) > 0:
        excluded_df.to_csv(args.excluded_rows_csv, index=False)
        print(f"Details written to {args.excluded_rows_csv}")
        print(excluded_df.head(10).to_string(index=False))

    n_with_drops = (features["n_dropped_points"] > 0).sum()
    print(f"\n{n_with_drops} retained row(s) had at least one individual point dropped "
          f"but were still usable via the remaining points in that group.")
    if n_with_drops > 0:
        print("Sample of point-level drops (see 'dropped_points' column for full detail):")
        print(features.loc[features["n_dropped_points"] > 0,
                            ["part_id", "dropped_points"]].head(10).to_string(index=False))

    if len(features) == 0:
        print("\nNo valid rows remain - check the excluded rows report above.")
        return

    labels_binary = (features["label"].str.lower() == "fail").astype(int)
    has_extent = "extent" in features.columns

    print("\nPer-part severity features (first 10 rows):")
    display_cols = [c for c in features.columns if c not in ("dropped_points",)]
    print(features[display_cols].head(10).to_string(index=False))

    crosstab_diagnostic(features, "truck_number", "well_size")
    within_truck_size_comparison(features, severity_col="dE_max")

    if standard_lab is not None:
        print("\n--- Head-to-head: which reference color better predicts your fail calls? ---")
        auc_base = roc_auc_score(labels_binary, features["dE_max"])
        auc_standard = roc_auc_score(labels_binary, features["dE_max_vs_standard"])
        print(f"dE_max (vs part's own base)  -> ROC AUC: {auc_base:.4f}")
        print(f"dE_max_vs_standard (vs spec) -> ROC AUC: {auc_standard:.4f}")

        print("\n--- Diagnostic: has each part's own base color drifted from the standard? ---")
        auc_base_drift = roc_auc_score(labels_binary, features["dE_base_vs_standard"])
        print(f"ROC AUC of dE_base_vs_standard predicting thermal-shock fail: {auc_base_drift:.4f}")
        print("(Near 0.5 is good here - means base-color drift isn't confounding the fail calls.)")

        analyze_group_confounding(
            features, args.group_col,
            ["dE_max", "dE_base_vs_standard"],
            min_group_size=args.min_group_size,
        )

        severity_col = "dE_max_vs_standard"
    else:
        severity_col = "dE_max"

    print(f"\n--- Univariate check: does {severity_col} alone separate pass/fail? ---")
    auc_dE = roc_auc_score(labels_binary, features[severity_col])
    print(f"ROC AUC using {severity_col} alone: {auc_dE:.4f}  (0.5 = chance, 1.0 = perfect)")

    best_thresh, sens, spec = find_best_threshold(features[severity_col], labels_binary)
    print(f"Best single {severity_col} threshold: {best_thresh:.2f}")
    print(f"  At this threshold -> sensitivity (fail recall): {sens:.3f}, specificity (pass recall): {spec:.3f}")

    print(f"\n--- CIE76 (plain Euclidean) vs CIEDE2000: does the formula choice matter here? ---")
    auc_76 = roc_auc_score(labels_binary, features["dE_max_cie76"])
    correlation_76_2000 = features["dE_max"].corr(features["dE_max_cie76"])
    print(f"dE_max (CIEDE2000) AUC: {auc_dE:.4f}")
    print(f"dE_max (CIE76)     AUC: {auc_76:.4f}")
    print(f"Correlation between the two formulas' dE_max values: {correlation_76_2000:.4f}")
    best_thresh_76, sens_76, spec_76 = find_best_threshold(features["dE_max_cie76"], labels_binary)
    print(f"Best single CIE76 threshold: {best_thresh_76:.2f} "
          f"(sensitivity={sens_76:.3f}, specificity={spec_76:.3f})")
    print("If the two AUCs are close and correlation is high, CIE76 is a safe, much simpler")
    print("substitute for a spreadsheet - just use its own threshold above, not 3.21 (that")
    print("number is specific to CIEDE2000 and will NOT transfer directly to CIE76 values).")

    if args.stratify_by is not None and args.stratify_by in features.columns:
        print(f"\n--- Per-{args.stratify_by} thresholds (instead of one pooled threshold) ---")
        for group_val, group_df in features.groupby(args.stratify_by):
            group_labels = (group_df["label"].str.lower() == "fail").astype(int)
            n_fail = int(group_labels.sum())
            n_total = len(group_df)

            if n_fail < 5 or n_fail == n_total:
                print(f"{args.stratify_by}={group_val}: {n_total} parts, {n_fail} fail(s) - "
                      f"too few fail examples to fit a reliable threshold, skipping")
                continue

            group_auc = roc_auc_score(group_labels, group_df[severity_col])
            g_thresh, g_sens, g_spec = find_best_threshold(group_df[severity_col], group_labels)
            print(f"{args.stratify_by}={group_val}: {n_total} parts, {n_fail} fail(s) "
                  f"({n_fail / n_total:.1%}) | AUC={group_auc:.4f} | "
                  f"threshold={g_thresh:.2f} | sensitivity={g_sens:.3f}, specificity={g_spec:.3f}")

    print(f"\n--- Does extent add anything beyond {severity_col} alone? ---")
    if not has_extent:
        print("No mask_pixels column found in this CSV - skipping extent analysis.")
        feature_cols = [severity_col, "op_agreement_gap"]
    else:
        print(f"Extent range in this dataset: {features['extent'].min():.4f} - {features['extent'].max():.4f}")
        auc_extent_alone = roc_auc_score(labels_binary, features["extent"])
        print(f"ROC AUC using extent alone: {auc_extent_alone:.4f}")
        feature_cols = [severity_col, "extent", "op_agreement_gap"]

    # op_agreement_gap can be NaN for rows where only one operator had a
    # usable point - drop those just for the multivariate model fit.
    model_df = features.dropna(subset=feature_cols)
    n_dropped_for_model = len(features) - len(model_df)
    if n_dropped_for_model > 0:
        print(f"\n({n_dropped_for_model} row(s) excluded from the multivariate model only, "
              f"due to missing op_agreement_gap - single-operator rows)")

    print("\n--- Multivariate check: severity" + (" + extent" if has_extent else "") + " + agreement ---")
    X = model_df[feature_cols].values
    y = (model_df["label"].str.lower() == "fail").astype(int).values
    print(f"Class balance in this data: {int((y == 0).sum())} pass, {int((y == 1).sum())} fail "
          f"({(y == 1).sum() / len(y):.1%} fail)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # class_weight="balanced" penalizes mistakes on the minority (fail) class
    # more heavily during training - without this, an imbalanced dataset like
    # yours (~19% fail) pushes the model toward under-predicting fail, since
    # predicting "pass" for everything already scores well on raw accuracy.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    auc_multi = roc_auc_score(y, cv_probs)
    print(f"Cross-validated ROC AUC: {auc_multi:.4f}")

    # Two views of the same model: the naive 0.5 cutoff, and the threshold
    # that best balances catching fails vs. false alarms (Youden's J, same
    # method used for the univariate severity threshold above). Comparing
    # both shows how much the fixed 0.5 cutoff alone was hurting fail recall.
    cv_preds_default = (cv_probs >= 0.5).astype(int)
    best_model_thresh, model_sens, model_spec = find_best_threshold(cv_probs, y)
    cv_preds_optimized = (cv_probs >= best_model_thresh).astype(int)

    print(f"\n--- At default 0.5 probability cutoff ---")
    print(classification_report(y, cv_preds_default, target_names=["pass", "fail"]))
    print("Confusion matrix (rows=true, cols=predicted), order: [pass, fail]")
    print(confusion_matrix(y, cv_preds_default))

    print(f"\n--- At optimized probability cutoff ({best_model_thresh:.3f}) ---")
    print(f"(sensitivity/fail recall: {model_sens:.3f}, specificity/pass recall: {model_spec:.3f})")
    print(classification_report(y, cv_preds_optimized, target_names=["pass", "fail"]))
    print("Confusion matrix (rows=true, cols=predicted), order: [pass, fail]")
    print(confusion_matrix(y, cv_preds_optimized))

    model.fit(X, y)
    print("\nFeature coefficients (fit on full data, for interpretation only):")
    for name, coef in zip(feature_cols, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    print("\nInterpretation notes:")
    print(f"- If {severity_col} alone gives AUC > ~0.9, a simple dE threshold may be all you need.")
    print("- If adding extent barely moves AUC above severity alone, that confirms area doesn't")
    print("  carry much independent signal - meaning pixel-accurate segmentation likely isn't needed.")
    print("- A high op_agreement_gap coefficient means operator disagreement is itself predictive -")
    print("  some 'fail' calls may be borderline cases where even your inspectors aren't consistent.")
    print("- Compare the 0.5-cutoff report to the optimized-cutoff report above: the gap between")
    print("  them shows how much fail recall was being lost to the imbalanced default threshold")
    print("  alone, separate from the class-weighting fix already applied during training.")


if __name__ == "__main__":
    main()
