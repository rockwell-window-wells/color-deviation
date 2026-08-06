"""
Converts LabelMe point annotations - created while looking at the CLAHE-
enhanced images from prepare_labeling_images.py - into training labels
that reference your RAW (non-CLAHE) images.

WHY THIS IS SAFE
    CLAHE only remaps pixel intensities - it never moves, crops, resizes,
    or pads anything. A click at (x, y) on the CLAHE version corresponds
    to the exact same (x, y) on the raw version, as long as both images
    have identical dimensions. This script enforces that: for every
    annotation, it checks the raw image's actual dimensions against what
    LabelMe recorded for the CLAHE image it was drawn on, and refuses to
    silently produce a mismatched label if they disagree (e.g. because a
    raw image was re-cropped after its CLAHE counterpart was made, or the
    wrong raw file matched by filename).

OUTPUT
    1. A clean manifest CSV (image_path, x, y, label) - the canonical,
       LabelMe-independent source of truth for point locations on your
       raw images.
    2. Optionally, YOLO-format bounding-box labels (each point expanded
       to a small fixed-size box) ready for training a detector, since
       most detection frameworks expect boxes, not raw points.

Usage:
    python convert_labelme_to_training_labels.py --labelme_dir clahe_for_labeling --raw_dir raw_cropped --output_dir training_labels

    # Also emit YOLO-format boxes (40x40px centered on each point):
    python convert_labelme_to_training_labels.py --labelme_dir clahe_for_labeling --raw_dir raw_cropped --output_dir training_labels --yolo_box_size 40

Requirements:
    pip install pillow pandas --break-system-packages
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def load_labelme_points(json_path):
    """
    Returns (image_filename, image_height, image_width, points) where
    points is a list of (x, y, label) tuples - only shape_type=="point"
    annotations are used; anything else is reported and skipped.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_filename = Path(data["imagePath"]).name  # strip any path LabelMe stored
    image_height = data["imageHeight"]
    image_width = data["imageWidth"]

    points = []
    skipped_non_point = 0
    for shape in data.get("shapes", []):
        if shape["shape_type"] != "point":
            skipped_non_point += 1
            continue
        x, y = shape["points"][0]
        points.append((x, y, shape["label"]))

    return image_filename, image_height, image_width, points, skipped_non_point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelme_dir", required=True,
                         help="Folder of LabelMe .json files, made against the CLAHE images")
    parser.add_argument("--raw_dir", required=True,
                         help="Folder of raw (non-CLAHE) images, same filenames as the CLAHE ones")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--yolo_box_size", type=int, default=None,
                         help="If set, also write YOLO-format labels with boxes of this size "
                              "(pixels) centered on each point")
    args = parser.parse_args()

    labelme_dir = Path(args.labelme_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(labelme_dir.glob("*.json"))
    print(f"Found {len(json_paths)} LabelMe annotation files in {labelme_dir}")

    rows = []
    n_dimension_mismatches = 0
    n_missing_raw = 0
    n_no_points = 0
    n_skipped_shapes_total = 0

    for json_path in json_paths:
        image_filename, clahe_h, clahe_w, points, skipped = load_labelme_points(json_path)
        n_skipped_shapes_total += skipped

        raw_path = raw_dir / image_filename
        if not raw_path.exists():
            print(f"WARNING: no raw image found for {json_path.name} "
                  f"(expected {raw_path}) - skipping")
            n_missing_raw += 1
            continue

        raw_w, raw_h = Image.open(raw_path).size
        if (raw_h, raw_w) != (clahe_h, clahe_w):
            print(f"WARNING: dimension mismatch for {image_filename} - "
                  f"CLAHE image was {clahe_w}x{clahe_h}, raw image is {raw_w}x{raw_h}. "
                  f"Coordinates would NOT transfer correctly - skipping this file.")
            n_dimension_mismatches += 1
            continue

        if not points:
            n_no_points += 1
            # A clean part with zero candidates is a legitimate, valuable
            # training example - record it with no rows rather than
            # silently dropping the image from the dataset entirely.
            rows.append({"image_path": str(raw_path), "x": None, "y": None, "label": None})
            continue

        for x, y, label in points:
            rows.append({"image_path": str(raw_path), "x": x, "y": y, "label": label})

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "point_labels.csv"
    manifest.to_csv(manifest_path, index=False)

    n_images_with_points = manifest[manifest["x"].notna()]["image_path"].nunique()
    n_images_no_points = manifest[manifest["x"].isna()]["image_path"].nunique()
    print(f"\nSaved {manifest_path}")
    print(f"  {n_images_with_points} image(s) with at least one candidate point")
    print(f"  {n_images_no_points} image(s) with zero candidate points (clean parts)")
    print(f"  {manifest['x'].notna().sum()} total point annotations")
    if n_dimension_mismatches:
        print(f"  {n_dimension_mismatches} file(s) skipped due to dimension mismatch - "
              f"check these manually, coordinates could not be trusted")
    if n_missing_raw:
        print(f"  {n_missing_raw} file(s) skipped - no matching raw image found")
    if n_skipped_shapes_total:
        print(f"  {n_skipped_shapes_total} non-point shape(s) skipped (only point "
              f"annotations are used)")

    if args.yolo_box_size:
        yolo_dir = output_dir / "yolo_labels"
        yolo_dir.mkdir(exist_ok=True)
        half = args.yolo_box_size / 2
        label_names = sorted(manifest["label"].dropna().unique())
        label_to_id = {name: i for i, name in enumerate(label_names)}

        for image_path, group in manifest.groupby("image_path"):
            img_w, img_h = Image.open(image_path).size
            stem = Path(image_path).stem
            lines = []
            for _, row in group.iterrows():
                if pd.isna(row["x"]):
                    continue  # clean part, no boxes - empty label file is correct YOLO convention
                cls_id = label_to_id[row["label"]]
                x_center = row["x"] / img_w
                y_center = row["y"] / img_h
                box_w = args.yolo_box_size / img_w
                box_h = args.yolo_box_size / img_h
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            (yolo_dir / f"{stem}.txt").write_text("\n".join(lines))

        classes_path = yolo_dir / "classes.txt"
        classes_path.write_text("\n".join(label_names))
        print(f"\nWrote YOLO-format labels to {yolo_dir} "
              f"({args.yolo_box_size}x{args.yolo_box_size}px boxes)")
        print(f"Classes: {label_names}")


if __name__ == "__main__":
    main()
