# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:07:06 2025

@author: Ryan.Larson
"""

import cv2
import os
import pandas as pd
import numpy as np

# --------------------------
# 1. Set folders
# --------------------------
ref_folder = r"G:\Shared drives\RockWell Shared\Engineering\Engineering Projects\DLFT\DLFT Testing\Color Checking\Validation Masks\Ryan's Masks"    # Reference images
test_folder = r"G:\Shared drives\RockWell Shared\Engineering\Engineering Projects\DLFT\DLFT Testing\Color Checking\Validation Masks\Duncan's Masks"  # Test images

# --------------------------
# 2. Prepare list of files
# --------------------------
ref_files = set(os.listdir(ref_folder))
test_files = set(os.listdir(test_folder))

common_files = sorted(ref_files & test_files)

# --------------------------
# 3. Function to calculate percentages
# --------------------------
def calc_white_pixel_overlap(ref_img_path, test_img_path):
    # Read images in grayscale
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to binary: consider pixels > 0 as white
    _, ref_bin = cv2.threshold(ref_img, 1, 255, cv2.THRESH_BINARY)
    _, test_bin = cv2.threshold(test_img, 1, 255, cv2.THRESH_BINARY)

    # Count total white pixels in ref
    ref_white_count = np.sum(ref_bin == 255)

    if ref_white_count == 0:
        return np.nan, np.nan  # Avoid division by zero

    # Pixels where both are white
    common_white_count = np.sum((ref_bin == 255) & (test_bin == 255))

    # White pixels in test not in ref
    extra_white_count = np.sum((test_bin == 255) & (ref_bin != 255))

    # Percentages
    perc_common = (common_white_count / ref_white_count) * 100
    perc_extra = (extra_white_count / ref_white_count) * 100

    return perc_common, perc_extra

# --------------------------
# 4. Process all common files
# --------------------------
results = []

for fname in common_files:
    ref_path = os.path.join(ref_folder, fname)
    test_path = os.path.join(test_folder, fname)

    perc_common, perc_extra = calc_white_pixel_overlap(ref_path, test_path)
    results.append({
        "filename": fname,
        "perc_common_white": perc_common,
        "perc_extra_white": perc_extra
    })

# --------------------------
# 5. Create DataFrame
# --------------------------
df = pd.DataFrame(results)

# --------------------------
# 6. Print results and summary
# --------------------------
print(df)

print("\nSummary statistics:")
print(df[["perc_common_white", "perc_extra_white"]].describe())
