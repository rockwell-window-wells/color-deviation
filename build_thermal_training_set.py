"""
Select and copy only the images that have a matching row in your severity
features CSV (produced by thermal_shock_severity_analysis.py --features_csv_out),
and build a manifest pairing each copied image to its dE_max target - ready
for training an image-based severity regression model.

Expected folder structure (same as before):
    images_root/
        6024_FAIL/
            image_00070.jpg
            ...
        6024_PASS/
            ...
        7024_FAIL/
        7024_PASS/
        ...

Usage:
    # Step 1 - export features with dE_max from the severity script:
    python thermal_shock_severity_analysis.py --csv Denali_Color_Data.csv --features_csv_out features.csv

    # Step 2 - match those rows to actual image files and copy them:
    python build_thermal_training_set.py --features_csv features.csv --images_root ./images --output_dir ./training_set

Output:
    output_dir/images/<original filename>   - copies of only the matched images
    output_dir/manifest.csv                 - image_filepath, part_id, well_size, label, dE_max, ...
    Console report of any CSV rows with no matching image, and any images
    found with no matching CSV row (both are worth checking, not silently
    dropping).
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PART_SIZE_PATTERN = re.compile(r"\d{4}")
LABEL_PATTERN = re.compile(r"(PASS|FAIL)", re.IGNORECASE)
IMAGE_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in the filename


def parse_subfolder_name(folder_name):
    part_size_match = PART_SIZE_PATTERN.search(folder_name)
    label_match = LABEL_PATTERN.search(folder_name)
    part_size = int(part_size_match.group(0)) if part_size_match else None
    label = label_match.group(0).upper() if label_match else None
    return part_size, label


def index_images(images_root):
    """
    Walks the size_LABEL subfolders and builds a lookup:
        (well_size, label, image_number) -> filepath
    Also flags subfolders that don't parse cleanly, same as the earlier
    folder-scanning script.
    """
    images_root = Path(images_root)
    lookup = {}
    skipped_folders = []
    skipped_files = []

    for subfolder in sorted(p for p in images_root.iterdir() if p.is_dir()):
        well_size, label = parse_subfolder_name(subfolder.name)
        if well_size is None or label is None:
            skipped_folders.append(subfolder.name)
            continue

        for img_path in sorted(subfolder.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            match = IMAGE_NUMBER_PATTERN.search(img_path.stem)
            if not match:
                skipped_files.append(str(img_path))
                continue
            image_number = int(match.group(0))
            lookup[(well_size, label, image_number)] = img_path

    return lookup, skipped_folders, skipped_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True,
                         help="Features CSV from thermal_shock_severity_analysis.py --features_csv_out")
    parser.add_argument("--images_root", required=True,
                         help="Parent folder containing the well_size_LABEL subfolders")
    parser.add_argument("--output_dir", required=True,
                         help="Where to copy matched images and write the manifest")
    parser.add_argument("--id_col", default="part_id",
                         help="Column in features_csv holding the image_file_number")
    args = parser.parse_args()

    features = pd.read_csv(args.features_csv)
    print(f"Loaded {len(features)} rows from {args.features_csv}")

    required_cols = {args.id_col, "well_size", "label"}
    missing = required_cols - set(features.columns)
    if missing:
        print(f"features_csv is missing required column(s): {missing}")
        print("Make sure you exported it with --features_csv_out from the severity script,")
        print("and that your original data had well_size/truck columns to carry through.")
        return

    image_lookup, skipped_folders, skipped_files = index_images(args.images_root)
    print(f"Indexed {len(image_lookup)} images across {args.images_root}")

    output_dir = Path(args.output_dir)
    images_out_dir = output_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    unmatched_csv_rows = []
    matched_image_keys = set()

    for _, row in features.iterrows():
        well_size = int(row["well_size"])
        label = str(row["label"]).upper()
        image_number = int(row[args.id_col])

        key = (well_size, label, image_number)
        img_path = image_lookup.get(key)

        if img_path is None:
            unmatched_csv_rows.append({
                "part_id": image_number, "well_size": well_size, "label": label,
            })
            continue

        matched_image_keys.add(key)
        dest_path = images_out_dir / img_path.name
        shutil.copy2(img_path, dest_path)

        manifest_row = {"image_filepath": str(dest_path)}
        manifest_row.update(row.to_dict())
        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nMatched and copied {len(manifest_rows)} of {len(features)} CSV rows to {images_out_dir}")
    print(f"Manifest written to {manifest_path}")

    if unmatched_csv_rows:
        unmatched_df = pd.DataFrame(unmatched_csv_rows)
        unmatched_path = output_dir / "unmatched_csv_rows.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"\n{len(unmatched_csv_rows)} CSV row(s) had no matching image file - "
              f"details in {unmatched_path}")
        print(unmatched_df.head(10).to_string(index=False))

    # Images that exist on disk but weren't referenced by any CSV row -
    # these just won't have a severity label, worth knowing about but not
    # necessarily a problem (e.g. images collected before/after this CSV's date range).
    all_image_keys = set(image_lookup.keys())
    unmatched_images = all_image_keys - matched_image_keys
    if unmatched_images:
        print(f"\n{len(unmatched_images)} image(s) on disk had no matching CSV row "
              f"(not copied, no severity data available for them)")

    if skipped_folders:
        print(f"\n{len(skipped_folders)} subfolder(s) couldn't be parsed for well_size/label:")
        for name in skipped_folders:
            print(f"  {name}")

    if skipped_files:
        print(f"\n{len(skipped_files)} image file(s) couldn't have a number extracted from their name")


if __name__ == "__main__":
    main()
