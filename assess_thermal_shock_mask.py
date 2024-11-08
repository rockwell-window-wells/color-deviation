# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:08:08 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_bounding_box(mask):
    # Find all non-zero points (the black regions in your case)
    coords = cv2.findNonZero(mask)  # returns a list of points [(x1, y1), (x2, y2), ...]
    
    # Get the bounding rectangle that fully contains the black region
    x, y, w, h = cv2.boundingRect(coords)
    
    return x, y, w, h

def overlay_masks(ref_mask, instance_mask):
    return ref_mask & instance_mask


if __name__ == "__main__":
    ref_mask_file = "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/6024_flats_mask.png"
    instance_mask_files = [
        "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/image_00008_THERMAL_SHOCK.png",
        "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/image_00009_THERMAL_SHOCK.png",
        "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/image_00012_THERMAL_SHOCK.png"
        ]
    
    for instance_mask_file in instance_mask_files:
        ref_mask = cv2.imread(ref_mask_file)
        instance_mask = cv2.imread(instance_mask_file)
        
        if len(ref_mask.shape) == 3:
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
            
        if len(instance_mask.shape) == 3:
            instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_BGR2GRAY)
            
        # Threshold to ensure binary values
        _, ref_mask = cv2.threshold(ref_mask, 127, 1, cv2.THRESH_BINARY)
        _, instance_mask = cv2.threshold(instance_mask, 127, 1, cv2.THRESH_BINARY)
        
        combined_mask = overlay_masks(ref_mask, instance_mask)
        # plt.imshow(combined_mask, cmap='gray', )
        # plt.show()
        # plt.close()
        
        # Get the count of white pixels in the combined mask
        npx_combined = np.count_nonzero(combined_mask)
        
        npx_ref = np.count_nonzero(ref_mask)
        
        pct_thermal_shock = npx_combined / npx_ref
        
        print(f'\n{instance_mask_file}:\n{np.around(100*pct_thermal_shock,2)}%')