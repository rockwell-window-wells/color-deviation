"""
Automatically estimate dE_max from a part image, with NO machine learning
involved - this is a deterministic pipeline, re-solved fresh for every
image with no parameters carried over between images or fit to any
training set. That's deliberate: every CNN approach we tried had the
opportunity to learn a truck-specific shortcut, because it was fitting
parameters to labeled examples that happen to cluster by truck. A
deterministic algorithm has nothing to fit and nothing to overfit to - the
same fixed logic runs identically on a truck it's never seen.

WHAT'S DIFFERENT FROM A NAIVE "READ PIXELS AS LAB" APPROACH
    Raw camera RGB is not physically comparable to true LAB unless you
    account for lighting - but critically, this algorithm sidesteps that
    problem rather than solving it head-on: it ONLY ever compares the
    baseline region to the worst region WITHIN THE SAME PHOTO, and both
    experience identical lighting by physical necessity (same shot, same
    moment). That means no cross-image color calibration is needed at all -
    unlike the CNN pipeline, which needed consistency ACROSS different
    photos and therefore needed a card-anchored correction. Applying that
    same cross-image correction here would be wrong: it assumes a photo's
    average content should be neutral gray, which is false for a crop
    that's mostly one deliberately-colored product surface, and would
    force the product's real color toward gray. This pipeline converts
    directly from the raw crop to LAB, then relies entirely on the within-
    image relative comparison for validity.

PIPELINE
    1. Sample FIXED baseline regions (known probe locations, not
       algorithmically inferred) and correct the whole image so they read
       as the true color standard - removes truck/session lighting
       variation at its source.
    2. Optionally (--no_background_model to disable) model any residual
       smooth shading gradient via a large-radius robust background model
       (Siril/PixInsight-DBE style), and measure each window against that
       local trend rather than the raw standard.
    3. Divide the valid region into windows and average LAB within each.
    4. Take a high percentile of each window's dE as the estimated
       dE_max - robust to a single noisy window (dirt, glare, a smudge).

    This replaced several earlier approaches (k-means clustering,
    median-of-all-windows, ring-based local contrast) that all inferred
    baseline algorithmically and turned out to be vulnerable to exactly
    the most severe cases - fixed, known probe locations sidestep that.

Usage:
    python auto_de_extraction.py --manifest preprocessed_images/manifest.csv --output_csv auto_de_results.csv --n_workers 20

Requirements:
    pip install pillow numpy pandas scikit-learn scipy --break-system-packages
"""

import os

# Must be set BEFORE numpy/scipy/sklearn are imported - caps each worker
# process to a single internal thread for its linear algebra operations.
# Without this, running N worker PROCESSES (each of which may also spawn
# its own BLAS threads) can oversubscribe the CPU far beyond its core
# count, which often makes a "parallel" run slower than sequential. With
# this cap, the parallelism comes cleanly from --n_workers alone.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

# ===========================================================================
# Real masks, verified: white=valid/flat (57-65% of the crop region across
# all four sizes, consistent with baseline material being the majority by
# design), black=exclude (ribs/curved edges). No inversion needed.
# ===========================================================================
FLAT_REGION_MASK_PATHS = {
    6024: "masks/reference_masks/6024_ref_mask.png",
    7024: "masks/reference_masks/7024_ref_mask.png",
    6038: "masks/reference_masks/6038_ref_mask.png",
    7038: "masks/reference_masks/7038_ref_mask.png",
}
MASK_INVERTED = False  # set True if black=keep, white=exclude in your files
MASK_THRESHOLD = 127   # pixel values above this (or below, if inverted) count as "valid"

_mask_cache = {}  # avoids re-loading/re-cropping the same mask file repeatedly


def load_flat_region_mask(well_size):
    """
    Loads the pre-made flat-region mask for this well_size, at full
    original resolution - no cropping. The masks already exclude
    non-product background (verified: for the ceiling-visible sizes,
    97-98% of the ceiling region is marked invalid) as well as ribs/curved
    edges, so a separate crop step is redundant. Returns a boolean array
    (True = valid/flat, safe to sample color from).
    """
    if well_size in _mask_cache:
        full_mask = _mask_cache[well_size]
    else:
        path = FLAT_REGION_MASK_PATHS.get(well_size)
        if path is None:
            return None
        try:
            mask_img = Image.open(path).convert("L")  # grayscale
        except FileNotFoundError:
            print(f"WARNING: mask file not found at '{path}' for well_size={well_size} - "
                  f"falling back to the darkness heuristic. Update FLAT_REGION_MASK_PATHS "
                  f"at the top of this script with your real file path.")
            _mask_cache[well_size] = None
            return None
        full_mask = np.array(mask_img)
        _mask_cache[well_size] = full_mask

    if full_mask is None:  # cached "file not found" result
        return None

    if MASK_INVERTED:
        valid = full_mask < MASK_THRESHOLD
    else:
        valid = full_mask >= MASK_THRESHOLD

    return valid




# ===========================================================================
# PLACEHOLDER - fill in with real (x, y, width, height) boxes per well_size,
# in original 3840x2160 image pixel coordinates, approximating where your
# physical LAB baseline probes are actually placed. A few locations per
# well_size (matching the "sides" of the part) is enough - these get
# pooled together to compute one correction per image.
# ===========================================================================
FIXED_BASELINE_REGIONS = {
    6024: [(100, 1280, 80, 80),
           (3660, 1280, 80, 80)],   # PLACEHOLDER
    7024: [(100, 1200, 80, 80),
           (3660, 1305, 80, 80)],   # PLACEHOLDER
    6038: [(100, 1245, 80, 80),
           (3660, 1275, 80, 80)],   # PLACEHOLDER
    7038: [(100, 1290, 80, 80),
           (3660, 1000, 80, 80)],   # PLACEHOLDER
}

# True color standard, converted from LAB (79.69, 1.75, 5.75) to RGB via
# LAB->XYZ->sRGB - same conversion used for GRAY_WORLD_TARGET in
# preprocess_thermal_images.py. This is what the baseline SHOULD read as;
# the correction scales each image so its own baseline reading matches
# this, which removes truck/session lighting variation (we found baseline
# brightness swinging from L=33 to L=71 across trucks) at its source,
# rather than trying to work around it downstream.
STANDARD_RGB = (205.6, 196.0, 186.9)
STANDARD_LAB = (79.69, 1.75, 5.75)

# Flag a correction as suspicious if any channel's scale factor is this
# extreme - could mean the baseline regions for that image are misplaced,
# occluded, or otherwise unreliable, worth knowing rather than silently
# trusting.
CORRECTION_SCALE_WARNING_THRESHOLD = 2.0

# Median-baseline design, validated directly against real ground truth
# (part 125: physical dE_max=7.72 landed almost exactly at the 99.5th
# percentile of per-window dE from a median-of-all-valid-windows baseline).
# WINDOW_SIZE is the box size in pixels used for averaging.
WINDOW_SIZE = 20

# Radius (in grid cells, i.e. multiples of WINDOW_SIZE) for the smooth
# background model - Siril/PixInsight-style gradient extraction, applied
# to the window grid. Deliberately large (hundreds of pixels) so any
# single defect patch barely moves it - this is what separates it from
# the earlier ring-based "local contrast" idea (which used a radius of
# only ~80px and ended up measuring edge steepness instead of true
# background level). MEDIAN (not mean) filtering makes it robust to a
# minority of window cells being genuinely defect-colored, the same way
# Siril's background sampling avoids being thrown off by stars/nebulae.
BACKGROUND_SMOOTHING_RADIUS = 21  # grid cells (~420px at WINDOW_SIZE=20)

# Percentile of per-window dE-from-standard values used as the estimate,
# not the single literal max - this is the "aggregated statistic" that
# makes the result robust to an isolated one-window artifact (dirt on the
# lens, a stray shadow, a smudge) a real inspector would ignore. Calibrated
# against one real example; worth re-checking against more once you have
# several to compare.
SEVERITY_PERCENTILE = 99.5

# Backstop against implausible per-window dE values (e.g. a single noisy
# window from glare or compression artifacts) - same sanity-ceiling
# principle used in thermal_shock_severity_analysis.py for the physical
# measurements. Real thermal shock in this data has topped out well below
# this; treat anything past it as a measurement artifact, not signal.
BLOCK_DE_SANITY_CEILING = 25.0
GROOVE_DARKNESS_THRESHOLD = 0.75  # a row is "groove shadow" if its median L is below this
                                    # fraction of the image's overall median L


def deltaE2000(lab1, lab2, kL=1, kC=1, kH=1):
    """Same CIEDE2000 formula validated throughout this project - kept
    identical here so estimated dE_max is directly comparable to your
    physically-measured dE_max."""
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

    return math.sqrt(
        (dLp / (kL * SL)) ** 2 + (dCp / (kC * SC)) ** 2 + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )


def srgb_to_lab(rgb_array):
    """
    Proper, non-approximated sRGB -> LAB conversion (sRGB -> linear RGB ->
    XYZ (D65) -> LAB). rgb_array: HxWx3 uint8. Returns HxWx3 float64 LAB.
    This is the calibration-dependent step - it only produces meaningful
    LAB values because the input has already been gray-world corrected
    against a known-true reference, not because this formula alone can
    infer true color from arbitrary camera RGB.
    """
    rgb = rgb_array.astype(np.float64) / 255.0

    # Inverse sRGB gamma -> linear RGB
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Linear RGB -> XYZ (sRGB primaries, D65 white point)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ matrix.T

    # Normalize by D65 reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xyz_n = xyz / np.array([Xn, Yn, Zn])

    delta = 6.0 / 29.0
    f = np.where(xyz_n > delta**3, np.cbrt(xyz_n), xyz_n / (3 * delta**2) + 4.0 / 29.0)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def mask_grooves(lab_image, darkness_threshold=GROOVE_DARKNESS_THRESHOLD):
    """
    Grooves show up as horizontal shadow bands (consistently darker rows),
    not as color - same exclusion your physical LAB measurements already
    apply by construction (probes aren't placed in the grooves). Returns a
    boolean mask, True = valid (non-groove) pixel.
    """
    L = lab_image[..., 0]
    row_medians = np.median(L, axis=1)
    overall_median = np.median(L)

    groove_rows = row_medians < (overall_median * darkness_threshold)
    mask = np.ones(L.shape, dtype=bool)
    mask[groove_rows, :] = False
    return mask


def compute_window_grid(lab_image, valid_mask, window_size=WINDOW_SIZE):
    """
    Averages LAB within window_size x window_size windows across the whole
    valid region, returned as a dense 2D grid. This is the only spatial
    decomposition the windowed-search design needs - no gradient map, no
    clustering, no connected components.
    Returns (window_lab_grid [rows,cols,3], window_valid_grid [rows,cols] bool).
    """
    h, w, _ = lab_image.shape
    y_starts = list(range(0, h - window_size + 1, window_size))
    x_starts = list(range(0, w - window_size + 1, window_size))
    n_rows, n_cols = len(y_starts), len(x_starts)

    window_lab_grid = np.full((n_rows, n_cols, 3), np.nan)
    window_valid_grid = np.zeros((n_rows, n_cols), dtype=bool)

    for by, y0 in enumerate(y_starts):
        for bx, x0 in enumerate(x_starts):
            window_mask = valid_mask[y0:y0 + window_size, x0:x0 + window_size]
            if window_mask.mean() < 0.5:  # mostly groove - skip
                continue
            window = lab_image[y0:y0 + window_size, x0:x0 + window_size, :]
            window_lab_grid[by, bx] = window[window_mask].mean(axis=0)
            window_valid_grid[by, bx] = True

    return window_lab_grid, window_valid_grid


def compute_baseline_correction(image_rgb, baseline_regions, target_rgb=STANDARD_RGB, valid_mask=None):
    """
    Samples RGB from the FIXED baseline regions (known probe locations,
    not algorithmically inferred) and computes the per-channel scale
    factor that would correct THIS image's own baseline reading to the
    true color standard. This removes truck/session lighting variation at
    its source - we found baseline brightness swinging from L=33 to L=71
    across trucks - rather than trying to infer a trustworthy baseline
    from image content, which every prior approach (clustering, median,
    ring-based) turned out to be vulnerable to on exactly the most severe
    (and most important) cases.

    valid_mask, if given, restricts sampling to pixels the flat-region
    mask actually considers valid within each box - so a baseline box
    that happens to clip a rib/groove edge doesn't let that darker
    material factor into the reading, same protection window sampling
    already has.

    Returns (scale [3], observed_baseline_rgb [3], n_pixels_sampled).
    """
    samples = []
    for (x, y, w, h) in baseline_regions:
        region = image_rgb[y:y + h, x:x + w, :]
        if valid_mask is not None:
            region_mask = valid_mask[y:y + h, x:x + w]
            region = region[region_mask]
        else:
            region = region.reshape(-1, 3)
        if len(region) > 0:
            samples.append(region)

    if not samples:
        raise ValueError("No valid (non-masked) pixels found in any FIXED_BASELINE_REGIONS box - "
                          "check that these regions actually fall within the flat-region mask.")

    all_samples = np.concatenate(samples, axis=0).astype(np.float64)
    observed = np.median(all_samples, axis=0)
    scale = np.array(target_rgb) / np.clip(observed, 1, 255)
    return scale, observed, len(all_samples)


def apply_correction(image_rgb, scale):
    corrected = np.clip(image_rgb.astype(np.float64) * scale, 0, 255).astype(np.uint8)
    return corrected


def estimate_smooth_background(window_lab_grid, window_valid_grid, radius=BACKGROUND_SMOOTHING_RADIUS):
    """
    Models the smooth, large-scale shading gradient across the part - the
    same kind of light-falloff effect astrophotography background-
    extraction tools (Siril, PixInsight DBE) remove before measuring faint
    objects sitting on top of it. Uses a large-radius MEDIAN filter over
    the window grid: large enough that a single defect patch barely
    shifts the local median (robust to it, the way Siril's sampling
    avoids stars/nebulae), while still tracking genuine gradual shading
    across the part.

    Gaps from invalid (groove-excluded) cells are filled with their
    nearest valid neighbor first, so the filter isn't corrupted by NaNs.
    Returns a background grid the same shape as window_lab_grid.
    """
    from scipy.ndimage import median_filter, distance_transform_edt

    filled = window_lab_grid.copy()
    if not window_valid_grid.all():
        invalid = ~window_valid_grid
        _, indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
        for c in range(3):
            filled[..., c] = window_lab_grid[..., c][tuple(indices)]

    background = np.empty_like(filled)
    for c in range(3):
        background[..., c] = median_filter(filled[..., c], size=radius, mode="nearest")
    return background


def estimate_severity(image_path, well_size, percentile=SEVERITY_PERCENTILE, use_background_model=True):
    """
    Estimates severity by:
      1. Sampling FIXED baseline regions (known probe locations, not
         inferred from image content) and computing the per-channel
         correction that maps this image's own baseline reading to the
         true color standard.
      2. Applying that correction to the whole image - this removes
         truck/session lighting variation at its source (baseline
         brightness was found to swing from L=33 to L=71 across trucks).
      3. If use_background_model=True: modeling any residual smooth
         shading gradient (Siril/PixInsight-DBE style) via a large-radius
         robust background model, and measuring each window against that
         LOCAL trend rather than the raw standard directly. If False:
         measuring each window directly against the standard, skipping
         this step entirely.

         WHY THIS IS TOGGLEABLE: real data showed dE_max_estimated
         collapsing toward (and past) the true value as true severity
         increases - a part in the top severity bin (5+) averaged
         dE_max_estimated LOWER than parts several bins milder, which is
         backwards. The background model is a suspected cause: a large
         defect could still pull even a big-radius median toward its own
         color on the most severe (usually largest-area) parts. Comparing
         both modes on the same data isolates whether this step is
         responsible before deciding whether to keep, shrink, or drop it.
      4. Taking a high percentile of those per-window deviations as the
         estimate.

    This replaces every prior baseline-INFERENCE approach (clustering,
    median-of-all-windows, ring-based local contrast) - all of them
    shared the same vulnerability: they assumed most of the valid region
    is normal material, which breaks down hardest on exactly the most
    severe cases, which is likely why every one of them showed an
    inverted or null result against real data despite different search
    strategies. Using known, fixed probe locations sidesteps that
    vulnerability entirely rather than patching it again.
    """
    if well_size not in FIXED_BASELINE_REGIONS:
        return {"dE_max_estimated": None,
                "reason": f"no FIXED_BASELINE_REGIONS defined for well_size={well_size}"}

    image = np.array(Image.open(image_path).convert("RGB"))

    # Mask is loaded BEFORE correction now, so baseline sampling can
    # respect it too. The darkness-heuristic fallback (mask_grooves) needs
    # a LAB image - it only uses ROW-RELATIVE darkness patterns, which a
    # uniform per-channel correction barely changes, so running it on the
    # raw (pre-correction) image is a safe, simpler ordering than
    # requiring the corrected image to exist first.
    valid_mask = load_flat_region_mask(well_size)
    used_real_mask = valid_mask is not None
    if not used_real_mask:
        valid_mask = mask_grooves(srgb_to_lab(image))

    scale, observed_baseline_rgb, n_baseline_pixels = compute_baseline_correction(
        image, FIXED_BASELINE_REGIONS[well_size], STANDARD_RGB, valid_mask=valid_mask
    )
    correction_suspicious = bool(np.any(scale > CORRECTION_SCALE_WARNING_THRESHOLD)
                                  or np.any(scale < 1.0 / CORRECTION_SCALE_WARNING_THRESHOLD))

    corrected_image = apply_correction(image, scale)
    lab_image = srgb_to_lab(corrected_image)

    window_lab_grid, window_valid_grid = compute_window_grid(lab_image, valid_mask)

    if window_valid_grid.sum() < 10:
        return {"dE_max_estimated": None, "reason": "too few valid windows after mask exclusion",
                "used_real_mask": used_real_mask, "valid_pixel_fraction": float(valid_mask.mean()),
                "n_blocks_dropped_as_implausible": 0}

    valid_labs = window_lab_grid[window_valid_grid]

    if use_background_model:
        # Smooth background model (Siril/PixInsight-DBE style) - captures
        # any residual gradual shading the 2-point fixed correction didn't
        # fully remove, without being an edge detector: the radius is
        # large enough (hundreds of pixels) that a real defect patch
        # barely moves it, so severity below measures true displacement
        # from local expected baseline, not edge steepness. See the
        # docstring above for why this is currently under test.
        background_grid = estimate_smooth_background(window_lab_grid, window_valid_grid)
        valid_background = background_grid[window_valid_grid]
        all_dEs = np.array([
            deltaE2000(tuple(bg), tuple(lab)) for bg, lab in zip(valid_background, valid_labs)
        ])
    else:
        # Compare directly to the standard - baseline should already read
        # as the standard after the fixed-region correction above, so no
        # further modeling step is applied.
        all_dEs = np.array([deltaE2000(STANDARD_LAB, tuple(lab)) for lab in valid_labs])

    # Drop individual windows whose dE is physically implausible (glare,
    # compression artifact, mask-edge bleed) before taking the percentile.
    plausible = all_dEs[all_dEs <= BLOCK_DE_SANITY_CEILING]
    n_dropped_as_implausible = len(all_dEs) - len(plausible)
    if len(plausible) == 0:  # everything got dropped - fall back rather than error
        plausible = all_dEs

    dE_estimate = float(np.percentile(plausible, percentile))

    return {
        "dE_max_estimated": dE_estimate,
        "used_real_mask": used_real_mask,
        "valid_pixel_fraction": float(valid_mask.mean()),
        "observed_baseline_rgb": tuple(observed_baseline_rgb),
        "n_baseline_pixels": n_baseline_pixels,
        "correction_scale": tuple(scale),
        "correction_suspicious": correction_suspicious,
        "used_background_model": use_background_model,
        "n_valid_windows": int(window_valid_grid.sum()),
        "n_blocks_dropped_as_implausible": int(n_dropped_as_implausible),
        "max_dE_raw": float(all_dEs.max()),
    }


def _process_one_row(task):
    """
    Runs estimate_severity for one manifest row and assembles the final
    result dict - must be a MODULE-LEVEL function (not nested inside
    main()) so it can be pickled and sent to worker processes.
    """
    row_index, image_filepath, well_size, percentile, part_id, dE_max_measured, label = task
    try:
        result = estimate_severity(image_filepath, well_size, percentile)
    except Exception as e:
        result = {"dE_max_estimated": None, "reason": str(e)}

    result["_row_index"] = row_index
    result["part_id"] = part_id
    result["image_filepath"] = image_filepath
    if dE_max_measured is not None:
        result["dE_max_measured"] = dE_max_measured
    if label is not None:
        result["label"] = label
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="CSV with image_filepath and well_size columns (e.g. from "
                              "build_thermal_training_set.py). If it also has dE_max (ground "
                              "truth from physical measurement), this script will report how "
                              "well the automated estimate correlates with it.")
    parser.add_argument("--output_csv", default="auto_de_results.csv")
    parser.add_argument("--percentile", type=float, default=SEVERITY_PERCENTILE)
    parser.add_argument("--progress_every", type=int, default=10,
                         help="Print a progress line every N images")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                         help="Write partial results to --output_csv every N images, so a "
                              "crash or interruption doesn't lose everything processed so far")
    parser.add_argument("--sample_n", type=int, default=None,
                         help="Only run on a random sample of N images from the manifest, "
                              "instead of all of them - useful for quickly checking whether a "
                              "parameter change helped before committing to a full run.")
    parser.add_argument("--sample_seed", type=int, default=42,
                         help="Random seed for --sample_n, so repeated runs with the same seed "
                              "sample the same rows (change it to get a different random subset)")
    parser.add_argument("--no_background_model", action="store_true",
                         help="Skip the Siril-style background model and compare each window "
                              "directly to the color standard instead. Use this to test whether "
                              "the background model is responsible for severity collapsing toward "
                              "(and past) the true value on your most severe parts - run once "
                              "with and once without this flag on the same data and compare.")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.sample_seed).reset_index(drop=True)
        print(f"Sampled {len(df)} random rows (seed={args.sample_seed}) for a quicker check")

    n_total = len(df)

    results = []
    n_failed = 0
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            result = estimate_severity(row["image_filepath"], int(row["well_size"]), args.percentile,
                                        use_background_model=not args.no_background_model)
        except Exception as e:
            result = {"dE_max_estimated": None, "reason": str(e)}
            n_failed += 1

        result["part_id"] = row.get("part_id")
        result["image_filepath"] = row["image_filepath"]
        result["well_size"] = row.get("well_size")
        if "truck_number" in row:
            result["truck_number"] = row["truck_number"]
        if "dE_max" in row:
            result["dE_max_measured"] = row["dE_max"]
        if "label" in row:
            result["label"] = row["label"]
        results.append(result)

        if i % args.progress_every == 0 or i == n_total:
            elapsed = time.time() - start_time
            rate = elapsed / i
            remaining = rate * (n_total - i)
            print(f"[{i}/{n_total}] {i/n_total:.0%} | "
                  f"{rate:.2f}s/image | "
                  f"elapsed {elapsed/60:.1f}min | "
                  f"est. remaining {remaining/60:.1f}min | "
                  f"{n_failed} failed so far")

        if i % args.checkpoint_every == 0 or i == n_total:
            pd.DataFrame(results).to_csv(args.output_csv, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")
    if n_failed:
        print(f"{n_failed} image(s) failed to process (see 'reason' column)")

    valid = results_df.dropna(subset=["dE_max_estimated"])
    print(f"\n{len(valid)} of {len(results_df)} images produced an estimate")

    if "used_real_mask" in valid.columns:
        n_real = int(valid["used_real_mask"].sum())
        n_fallback = len(valid) - n_real
        print(f"{n_real} used your real flat-region mask, {n_fallback} fell back to the "
              f"darkness heuristic (missing/unset FLAT_REGION_MASK_PATHS for that well_size)")
        if n_fallback > 0:
            print("Check FLAT_REGION_MASK_PATHS at the top of this script if that's unexpected.")

    if "valid_pixel_fraction" in valid.columns:
        frac = valid["valid_pixel_fraction"]
        print(f"Valid (non-excluded) pixel fraction: mean={frac.mean():.2f}, "
              f"range={frac.min():.2f}-{frac.max():.2f}")
        print("If this is very low (<0.1) or very high (>0.95) across most images, your mask's")
        print("black/white convention may be inverted - try setting MASK_INVERTED = True.")

    if "correction_suspicious" in valid.columns:
        n_suspicious = int(valid["correction_suspicious"].sum())
        if n_suspicious > 0:
            print(f"\n{n_suspicious} of {len(valid)} image(s) needed an unusually large baseline "
                  f"correction (a channel scale factor beyond {CORRECTION_SCALE_WARNING_THRESHOLD}x) - "
                  f"worth spot-checking these, since it could mean FIXED_BASELINE_REGIONS is "
                  f"misplaced, occluded, or otherwise unreliable for that well_size/image.")

    if "dE_max_measured" in valid.columns:
        corr = valid["dE_max_estimated"].corr(valid["dE_max_measured"])
        print(f"\nCorrelation between estimated and measured dE_max: {corr:.4f}")
        print("(This is the key number - it tells you whether this automatic pipeline is")
        print(" tracking your physically-measured ground truth, without any labeled training.)")

    if "label" in valid.columns:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        labels_binary = (valid["label"].str.upper() == "FAIL").astype(int)
        n_fail, n_pass = int(labels_binary.sum()), int((1 - labels_binary).sum())
        if labels_binary.nunique() > 1:
            auc_severity = roc_auc_score(labels_binary, valid["dE_max_estimated"])
            print(f"\nROC AUC of estimated dE_max predicting fail: {auc_severity:.4f} "
                  f"(n={len(valid)}: {n_fail} fail, {n_pass} pass)")
            print("(Compare this to 0.86, the AUC of your physically-measured dE_max, to see")
            print(" how much estimation accuracy this automated pipeline is giving up, if any.)")
            if n_fail < 30:
                # Rough Hanley-McNeil style standard error estimate - gives a sense of how
                # much this AUC could plausibly move with more data, not an exact CI.
                se = ((auc_severity * (1 - auc_severity)) / min(n_fail, n_pass)) ** 0.5
                print(f"NOTE: only {n_fail} fail examples in this sample - AUC here has a rough")
                print(f"      standard error of around +/-{se:.2f}. Treat this as noisy until you")
                print(f"      run a larger sample or the full dataset.")

            if "well_size" in valid.columns and valid["well_size"].nunique() > 1:
                print("\n--- Per-well_size breakdown (small groups are noisy - read counts, not just AUC) ---")
                for size, group in valid.groupby("well_size"):
                    g_labels = (group["label"].str.upper() == "FAIL").astype(int)
                    g_fail, g_pass = int(g_labels.sum()), int((1 - g_labels).sum())
                    if g_labels.nunique() > 1:
                        g_auc = roc_auc_score(g_labels, group["dE_max_estimated"])
                        print(f"  well_size={size}: n={len(group)} ({g_fail} fail, {g_pass} pass) -> AUC={g_auc:.4f}")
                    else:
                        print(f"  well_size={size}: n={len(group)} ({g_fail} fail, {g_pass} pass) -> "
                              f"only one class present, AUC not computable")

            if "defect_fraction_gradient_filtered" in valid.columns:
                auc_extent = roc_auc_score(labels_binary, valid["defect_fraction_gradient_filtered"])
                print(f"\nROC AUC of defect area fraction ALONE predicting fail: {auc_extent:.4f}")
                print("(Tests your hypothesis directly: does % of high-contrast area separate")
                print(" pass/fail on its own, independent of severity?)")

                mean_fail = valid.loc[labels_binary == 1, "defect_fraction_gradient_filtered"].mean()
                mean_pass = valid.loc[labels_binary == 0, "defect_fraction_gradient_filtered"].mean()
                print(f"Mean defect area fraction - FAIL parts: {mean_fail:.1%}, "
                      f"PASS parts: {mean_pass:.1%}")
                print("(Compare this gap to what you recalled from your earlier manual masking")
                print(" work - roughly 30% for fails vs 20-25% for passes.)")

                # Does extent add anything BEYOND severity, or is it redundant with it?
                X = valid[["dE_max_estimated", "defect_fraction_gradient_filtered"]].values
                y = labels_binary.values
                model = LogisticRegression(class_weight="balanced", max_iter=1000)
                model.fit(X, y)
                combined_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
                print(f"\nCombined (severity + extent) AUC: {combined_auc:.4f}")
                print(f"  vs severity alone: {auc_severity:.4f}")
                print("If combined is meaningfully higher than severity alone, extent is adding")
                print("real incremental information - worth keeping as a second feature, not just")
                print("a proxy for severity. If it's barely different, extent is likely redundant")
                print("with severity even though it may be predictive on its own above.")
            else:
                print("\n(No extent/area measurement in this run - the windowed-search design")
                print(" doesn't produce one, unlike the earlier clustering-based version. This")
                print(" is a deliberate tradeoff for simplicity; if extent turns out to matter,")
                print(" it would need to be added back deliberately, not as a side effect.)")


if __name__ == "__main__":
    main()
