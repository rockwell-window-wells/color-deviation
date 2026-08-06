"""
Build a CSV index from a folder of subfolders named like:

    6024_FAIL/
    6024_PASS/
    6038_FAIL/
    6038_PASS/
    ...

Each subfolder is expected to contain image files. For every image found,
this writes one row to the output CSV with:
    image_number  - the number pulled out of the image's filename
    part_size     - the 4-digit part size pulled from the subfolder name
    label         - PASS or FAIL, pulled from the subfolder name
    filename      - original image filename (for traceability)
    filepath      - full path to the image (for traceability)

Usage:
    python build_image_index.py --root_dir "C:/path/to/parent_folder" --output image_index.csv

Assumptions (flag these to me if they don't match your data):
    - Subfolder names contain a 4-digit part size and the word PASS or FAIL,
      separated by an underscore (e.g. "6024_FAIL"). Order/case doesn't matter.
    - Each image filename contains a number that identifies it (e.g.
      "IMG_0047.jpg", "0047.jpg", "part_47.png" all work) - the LAST run of
      digits in the filename is taken as the image number. If your naming
      convention puts the meaningful number somewhere else, let me know and
      I'll adjust the extraction logic.
"""

import argparse
import csv
import re
from pathlib import Path


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PART_SIZE_PATTERN = re.compile(r"\d{4}")
LABEL_PATTERN = re.compile(r"(PASS|FAIL)", re.IGNORECASE)
IMAGE_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")  # last run of digits in the string


def parse_subfolder_name(folder_name):
    """Extract part_size and label from a subfolder name like '6024_FAIL'."""
    part_size_match = PART_SIZE_PATTERN.search(folder_name)
    label_match = LABEL_PATTERN.search(folder_name)

    part_size = part_size_match.group(0) if part_size_match else None
    label = label_match.group(0).upper() if label_match else None

    return part_size, label


def extract_image_number(filename_stem):
    """Pull the last run of digits out of a filename (without extension)."""
    match = IMAGE_NUMBER_PATTERN.search(filename_stem)
    return match.group(0) if match else None


def build_index(root_dir):
    root_dir = Path(root_dir)
    rows = []
    skipped_folders = []
    skipped_files = []

    for subfolder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        part_size, label = parse_subfolder_name(subfolder.name)

        if part_size is None or label is None:
            skipped_folders.append(subfolder.name)
            continue

        for img_path in sorted(subfolder.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            image_number = extract_image_number(img_path.stem)
            if image_number is None:
                skipped_files.append(str(img_path))
                continue

            rows.append({
                "image_number": image_number,
                "part_size": part_size,
                "label": label,
                "filename": img_path.name,
                "filepath": str(img_path),
            })

    return rows, skipped_folders, skipped_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Parent folder containing the labeled subfolders")
    parser.add_argument("--output", default="image_index.csv", help="Output CSV path")
    args = parser.parse_args()

    rows, skipped_folders, skipped_files = build_index(args.root_dir)

    if not rows:
        print("No matching images found - check --root_dir and your folder naming.")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_number", "part_size", "label", "filename", "filepath"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")

    if skipped_folders:
        print(f"\nSkipped {len(skipped_folders)} subfolder(s) - couldn't find a 4-digit part size and PASS/FAIL:")
        for name in skipped_folders:
            print(f"  {name}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} file(s) - couldn't find a number in the filename:")
        for name in skipped_files[:20]:
            print(f"  {name}")
        if len(skipped_files) > 20:
            print(f"  ...and {len(skipped_files) - 20} more")


if __name__ == "__main__":
    main()
