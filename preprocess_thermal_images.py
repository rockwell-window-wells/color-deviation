"""
Preprocess thermal-shock part images before training/inference:
    1. Crop using a well_size-specific box (removes the size-varying amount
       of background/ceiling each well_size currently includes).
    2. Pad the crop to a square aspect ratio (prevents geometric distortion
       when resizing - see the aspect-ratio discussion from our conversation).
    3. Resize to one common size for the network.
    4. Apply per-channel gray-world color correction, anchored to TRUE target
       values measured from your Pantone reference card (not an arbitrary
       "average of my images" target) - corrects the white-balance/exposure
       drift between shop-floor locations.

===========================================================================
PLACEHOLDER VALUES - YOU MUST REPLACE THESE BEFORE USING THIS FOR REAL WORK
===========================================================================

CROP_BOXES: one box per well_size, in ORIGINAL IMAGE PIXEL COORDINATES
(not fractional/normalized). To find real values: open a representative
image from each well_size in an image viewer that shows pixel coordinates
(GIMP, Photoshop, even just hovering in most OS image viewers), and note
the (x, y) of the crop box's top-left corner and its width/height, adjusted
so the part is fully contained with a small margin, excluding as much
ceiling/background as reasonably possible.

GRAY_WORLD_TARGETS: the TRUE (R, G, B) values your Pantone card's neutral
gray patch should read as, under ideal/reference conditions. Get this by
photographing the card under your best, most "normal" lighting setup once,
cropping to just the neutral patch, and averaging its pixel values - that
becomes your fixed calibration target, used to correct every other photo's
color cast regardless of that shot's actual lighting.

Usage:
    python preprocess_thermal_images.py --manifest training_set/manifest.csv --output_dir ./preprocessed

Requirements:
    pip install pillow numpy pandas opencv-python --break-system-packages
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ===========================================================================
# Wider region including more baseline/normal-colored material on both
# sides of the thermal-shock area (same box for all well_size groups).
# ===========================================================================
CROP_BOXES = {
    6024: (0, 810, 3840, 1350),
    7024: (0, 810, 3840, 1350),
    6038: (0, 810, 3840, 1350),
    7038: (0, 810, 3840, 1350),
}

# ===========================================================================
# Your ACTUAL COLOR STANDARD (LAB 79.69, 1.75, 5.75), converted to RGB via
# LAB->XYZ->sRGB. This replaces an earlier version of this target that used
# the Pantone card's NEUTRAL GRAY patch instead - that was a conceptual
# bug, not just a different number: gray-world correction scales a photo's
# content to match its target, which only makes sense if the target
# represents what that content SHOULD look like. The neutral gray patch is
# not what your cream/tan product should look like, so using it as the
# target was forcing every corrected image's color toward gray. The color
# standard is the physically correct target - it's your product's actual
# expected color, so correcting toward it removes lighting cast without
# destroying the real hue.
#
# One tradeoff worth knowing: this will also homogenize genuine part-to-
# part/batch base-color drift toward the standard, since every image's
# baseline gets pulled toward the same target. Earlier analysis found that
# kind of drift was itself mostly a truck-level artifact rather than real
# severity signal, so removing it is more likely to help than hurt - but
# it means this correction is not "neutral" with respect to what
# information survives into the corrected image, worth remembering if you
# ever want to study base-color drift itself again later.
# ===========================================================================
GRAY_WORLD_TARGET = (205.6, 196.0, 186.9)

TARGET_WIDTH = 512   # long edge - matches the ~2:1 aspect ratio of the example crops
TARGET_HEIGHT = 256  # if your well_size crop boxes AREN'T all ~2:1, resizing straight to
                      # this shape will introduce a small stretch for the ones that differ -
                      # worth keeping your custom regions at a consistent aspect ratio if you
                      # can, so every well_size resizes into this shape without distortion

# CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters.
# Unlike a global contrast stretch, CLAHE enhances LOCAL contrast within
# small tiles - a much better match for this data, since the thermal-shock
# splotches are local/high-frequency while the curvature-driven lighting
# gradient across the part is smooth/low-frequency. CLAHE tends to leave
# that slow gradient largely alone while boosting the local variation we
# actually want more visible.
CLAHE_CLIP_LIMIT = 2.0        # higher = more contrast enhancement, but more noise amplification too
CLAHE_TILE_GRID_SIZE = (8, 8)  # number of tiles across the image; tile size scales with crop size


def apply_clahe(image, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """
    Applies CLAHE independently to each of R, G, B - not just a brightness/
    Value channel - since the manual GIMP experiment that motivated this
    used adjustments across Value, Green, and Blue, suggesting the useful
    signal isn't purely lightness.
    """
    arr = np.array(image)  # HxWx3, uint8
    channels = cv2.split(arr)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_channels = [clahe.apply(ch) for ch in channels]

    enhanced = cv2.merge(enhanced_channels)
    return Image.fromarray(enhanced)


def gray_world_correct(image, target_rgb, robust=True, clip_percentile=(10, 90)):
    """
    Per-channel gray-world correction: scales each channel so the image's
    own statistic (median, or a trimmed mean if robust=True) matches
    `target_rgb`, then clips to valid range.

    Using a robust statistic instead of the plain mean matters here because
    the thermal-shock area is a real, meaningful color outlier within the
    image - a plain mean would be pulled by it, which means the "correction"
    would partly be reacting to the defect itself rather than purely to
    lighting. A robust statistic (trimmed mean/median) is dominated by the
    majority-normal part surface instead, so the correction targets lighting
    without erasing the severity signal.
    """
    arr = np.array(image).astype(np.float32)

    corrected_channels = []
    for c in range(3):
        channel = arr[:, :, c]
        if robust:
            lo, hi = np.percentile(channel, clip_percentile)
            trimmed = channel[(channel >= lo) & (channel <= hi)]
            current_stat = np.median(trimmed) if trimmed.size > 0 else np.median(channel)
        else:
            current_stat = np.mean(channel)

        if current_stat < 1e-6:
            scale = 1.0
        else:
            scale = target_rgb[c] / current_stat

        corrected = np.clip(channel * scale, 0, 255)
        corrected_channels.append(corrected)

    corrected_arr = np.stack(corrected_channels, axis=2).astype(np.uint8)
    return Image.fromarray(corrected_arr)


def process_image(image_path, well_size, crop_boxes=CROP_BOXES, target_rgb=GRAY_WORLD_TARGET,
                   target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, no_resize=False):
    if well_size not in crop_boxes:
        raise ValueError(f"No crop box defined for well_size={well_size}. "
                          f"Add it to CROP_BOXES at the top of this script.")

    image = Image.open(image_path).convert("RGB")
    x, y, w, h = crop_boxes[well_size]
    cropped = image.crop((x, y, x + w, y + h))

    # Color-correct BEFORE resizing - this is important, not just ordering
    # for its own sake. Gray-world correction needs its statistics computed
    # from real image content at native resolution, not after any resampling
    # that could shift pixel value distributions.
    corrected = gray_world_correct(cropped, target_rgb)

    # CLAHE also runs at native resolution, before any resize - local tile
    # boundaries should align with real pixel content, not resampled blur.
    enhanced = apply_clahe(corrected)

    if no_resize:
        # Preserve native crop resolution - keeps area/extent, edge
        # smoothness, and shape/contrast detail fully intact, at the cost
        # of every image being large and (if crop_boxes vary by well_size)
        # not necessarily uniform in size. The training script handles
        # variable sizes via per-batch padding.
        return enhanced

    # No padding step - resizing straight to (target_width, target_height)
    # uses every output pixel for real content instead of burning half of
    # them on gray filler, PROVIDED your crop_boxes share roughly this
    # aspect ratio. A crop box with a very different ratio than
    # target_width:target_height will get mildly stretched by this resize -
    # keep your custom regions consistent in shape to avoid that entirely.
    resized = enhanced.resize((target_width, target_height), Image.LANCZOS)
    return resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="manifest.csv from build_thermal_training_set.py")
    parser.add_argument("--output_dir", required=True, help="Where to save preprocessed images")
    parser.add_argument("--target_width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target_height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--no_resize", action="store_true",
                         help="Keep crops at native resolution instead of resizing - preserves "
                              "defect area/extent, edge smoothness, and shape detail that "
                              "downsampling can blur away. Images will be large and possibly "
                              "non-uniform in size; the classifier training script pads to match "
                              "within each batch rather than requiring exact consistency.")
    parser.add_argument("--updated_manifest_out", default=None,
                         help="Path to write a new manifest pointing at the preprocessed images "
                              "(defaults to <output_dir>/manifest.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows from {args.manifest}")

    missing_sizes = set(df["well_size"].unique()) - set(CROP_BOXES.keys())
    if missing_sizes:
        print(f"WARNING: no crop box defined for well_size(s) {missing_sizes} - "
              f"these rows will be skipped. Add them to CROP_BOXES at the top of this script.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = []
    n_failed = 0

    for _, row in df.iterrows():
        well_size = int(row["well_size"])
        if well_size not in CROP_BOXES:
            continue

        try:
            processed = process_image(row["image_filepath"], well_size,
                                       target_width=args.target_width,
                                       target_height=args.target_height,
                                       no_resize=args.no_resize)
        except Exception as e:
            print(f"Failed to process {row['image_filepath']}: {e}")
            n_failed += 1
            continue

        out_path = output_dir / Path(row["image_filepath"]).name
        processed.save(out_path)

        new_row = row.to_dict()
        new_row["image_filepath"] = str(out_path)
        processed_rows.append(new_row)

    processed_df = pd.DataFrame(processed_rows)
    manifest_out = args.updated_manifest_out or str(output_dir / "manifest.csv")
    processed_df.to_csv(manifest_out, index=False)

    print(f"\nProcessed {len(processed_rows)} of {len(df)} images to {output_dir}")
    if n_failed:
        print(f"{n_failed} image(s) failed to process (see errors above)")
    print(f"Updated manifest written to {manifest_out}")
    print("\nRemember: CROP_BOXES and GRAY_WORLD_TARGET at the top of this script are still")
    print("placeholders - replace them with real measured values before trusting this output.")


if __name__ == "__main__":
    main()
