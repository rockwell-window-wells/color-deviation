# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:32:58 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2

def combine_masks(file_list):
    combined_mask = None
    for mask_file in file_list:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask & mask
    return combined_mask


if __name__ == "__main__":
    
    file_list = ['C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_1.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_2.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_3.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_4.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_5.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_6.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_7.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_8.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_9.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_10.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_11.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_12.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_13.png',
                 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_14.png',
                 ]
    
    result_mask = combine_masks(file_list)
    cv2.imwrite('combined_mask.png', result_mask)