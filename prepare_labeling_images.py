"""
Crops your original photos to a fixed region, then generates CLAHE-
enhanced copies of the CROPPED images purely for labeling in LabelMe.
The model itself trains on the cropped RAW images (see
convert_labelme_to_training_labels.py) - CLAHE is a labeling aid only.

WHY CROP HAPPENS FIRST, ONCE, BEFORE CLAHE
    Cropping is a GEOMETRIC transform - it changes the coordinate system,
    unlike CLAHE which only remaps pixel intensities in place. Doing the
    crop once and applying CLAHE to that same cropped array (rather than
    cropping the raw and CLAHE versions separately, or CLAHE-ing the
    original) guarantees both outputs share the exact same dimensions and
    coordinate system with no offset math required - a click at (x, y) on
    the cropped+CLAHE image is at that exact same (x, y) on the cropped
    raw image, full stop.

Usage:
    python prepare_labeling_images.py --input_dir original_photos --crop_box 900 0 2160 2160 --raw_output_dir raw_cropped --clahe_output_dir clahe_for_labeling

    --crop_box takes x y width height, in original-image pixel coordinates.

Requirements:
    pip install opencv-python-headless pillow numpy --break-system-packages
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def apply_clahe(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE on the L channel of LAB (standard approach - avoids
    distorting hue/saturation, only remaps local lightness contrast).
    Returns a BGR image, same dimensions as input.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def crop_image(image, crop_box):
    """
    crop_box is (x, y, width, height) in the ORIGINAL image's pixel
    coordinates. Returns the cropped PIL Image. Raises a clear error
    rather than silently clipping if the box doesn't fit - a crop that
    silently shrinks would break the fixed-dimension guarantee this whole
    workflow depends on.
    """
    x, y, w, h = crop_box
    img_w, img_h = image.size
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(
            f"crop_box {crop_box} doesn't fit inside a {img_w}x{img_h} image "
            f"(would need x+w<={img_w} and y+h<={img_h})"
        )
    return image.crop((x, y, x + w, y + h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                         help="Folder of original (uncropped) photos")
    parser.add_argument("--crop_box", type=int, nargs=4, required=True,
                         metavar=("X", "Y", "WIDTH", "HEIGHT"),
                         help="Crop region in original-image pixel coordinates")
    parser.add_argument("--raw_output_dir", required=True,
                         help="Where to save cropped (non-CLAHE) images - this is what "
                              "the model actually trains on")
    parser.add_argument("--clahe_output_dir", required=True,
                         help="Where to save cropped+CLAHE images, for opening in LabelMe")
    parser.add_argument("--clip_limit", type=float, default=2.0)
    parser.add_argument("--tile_grid_size", type=int, default=8,
                         help="CLAHE tile grid is this x this")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_output_dir = Path(args.raw_output_dir)
    clahe_output_dir = Path(args.clahe_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    clahe_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Cropping to box (x, y, w, h) = {tuple(args.crop_box)}")

    n_written = 0
    n_failed = 0
    for path in image_paths:
        try:
            original = Image.open(path).convert("RGB")
            cropped = crop_image(original, args.crop_box)
        except ValueError as e:
            print(f"WARNING: skipping {path.name} - {e}")
            n_failed += 1
            continue

        # Save the cropped RAW image first - this is the actual training input.
        raw_out_path = raw_output_dir / path.name
        cropped.save(raw_out_path)

        # CLAHE is applied to this SAME cropped array, so dimensions match exactly.
        cropped_bgr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        enhanced_bgr = apply_clahe(cropped_bgr, args.clip_limit,
                                    (args.tile_grid_size, args.tile_grid_size))
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enhanced_image = Image.fromarray(enhanced_rgb)

        assert enhanced_image.size == cropped.size, \
            f"Dimension mismatch for {path.name} - this should never happen with CLAHE"

        clahe_out_path = clahe_output_dir / path.name
        enhanced_image.save(clahe_out_path)

        n_written += 1

    print(f"\nWrote {n_written} image pairs "
          f"({args.crop_box[2]}x{args.crop_box[3]} each) to:")
    print(f"  {raw_output_dir}  (cropped raw - trains the model)")
    print(f"  {clahe_output_dir}  (cropped + CLAHE - open this folder in LabelMe)")
    if n_failed:
        print(f"{n_failed} image(s) skipped - crop_box didn't fit, check their dimensions")
    print(f"\nWhen labeling is done, run convert_labelme_to_training_labels.py with "
          f"--labelme_dir {clahe_output_dir} --raw_dir {raw_output_dir}")


if __name__ == "__main__":
    main()
