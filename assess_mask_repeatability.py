# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 07:37:48 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2
import tkinter.filedialog as fd
from tkinter import Tk
import os

def get_white_count(maskfile, ref_mask):
    mask = cv2.imread(maskfile)
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask & ref_mask
    white_count = np.count_nonzero(binary_mask)
    return white_count


if __name__ == "__main__":
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()  # Hide the main window
    directory = fd.askdirectory(title="Select the Mask Directory")
    root.destroy()
    
    ref_mask_file = "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/6024_flats_mask.png"
    ref_mask = cv2.imread(ref_mask_file)
    _, ref_mask = cv2.threshold(ref_mask, 127, 1, cv2.THRESH_BINARY)
    
    maskfiles = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
    
    white_counts = {}
    for maskfile in maskfiles:
        white_counts[os.path.basename(maskfile)] = get_white_count(maskfile, ref_mask)
        
    white_vals = list(white_counts.values())
    std_white = np.std(white_vals, ddof=1)
    mean_white = np.mean(white_vals)
    
    CV = std_white / mean_white   # Coefficient of variation ()
    SE = std_white / np.sqrt(len(white_vals))