# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:42:26 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2
import tkinter.filedialog as fd
from tkinter import Tk
import os
import pandas as pd
from scipy import stats

def get_white_count(maskfile, binary_ref_mask):
    mask = cv2.imread(maskfile)
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask & binary_ref_mask
    white_count = np.count_nonzero(binary_mask)
    return white_count


if __name__ == "__main__":
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    masks_directory = fd.askdirectory(title='Select the Masks Directory')
    root.destroy()
    
    directories = [f.path for f in os.scandir(masks_directory) if f.is_dir() and f.name != 'reference_masks']
    
    # Iterate through the directories and process the files according to part size
    white_counts = []
    white_percents = []
    file_names = []
    pass_fails = []
    part_sizes = []
    for directory in directories:
        # Determine the part size
        if '6024' in directory:
            part_size = '6024'
        elif '6038' in directory:
            part_size = '6038'
        elif '7024' in directory:
            part_size = '7024'
        elif '7038' in directory:
            part_size = '7038'
        else:
            raise ValueError('An unexpected directory was found')
            
        if 'PASS' in directory:
            pass_fail = True
        else:
            pass_fail = False
        
        # Get the matching ref_mask ask a binary mask
        ref_mask_file = masks_directory + f'/reference_masks/{part_size}_ref_mask.png'
        ref_mask = cv2.imread(ref_mask_file)
        _, binary_ref_mask = cv2.threshold(ref_mask, 127, 1, cv2.THRESH_BINARY)
        ref_mask_white_count = np.count_nonzero(binary_ref_mask)
        
        # Process the masks
        for file in os.listdir(directory):
            mask_file = os.fsdecode(file)
            mask_file = os.path.join(directory, mask_file)
            file_names.append(os.path.basename(mask_file))
            white_count = get_white_count(mask_file, binary_ref_mask)
            white_counts.append(white_count)
            white_percent = white_count / ref_mask_white_count
            white_percents.append(white_percent)
            pass_fails.append(pass_fail)
            part_sizes.append(part_size)
        
            
    data = {'file': file_names,
            'part_size': part_sizes,
            'pass': pass_fails,
            'white_count': white_counts,
            'white_percent': white_percents}
        
    df = pd.DataFrame(data)
    
    
    # Statistical comparison
    pass_true = df[df['pass'] == True]['white_percent']
    pass_false = df[df['pass'] == False]['white_percent']
    
    mean_pass_true = pass_true.mean()
    mean_pass_false = pass_false.mean()
    
    mean_diff = mean_pass_true - mean_pass_false
    
    t_stat, p_value = stats.ttest_ind(pass_true, pass_false, equal_var=False)
    
    # Calculate the confidence interval for the difference in means
    confidence_level = 0.95  # 95% confidence interval
    degrees_freedom = len(pass_true) + len(pass_false) - 2
    stderr = np.sqrt(pass_true.var(ddof=1)/len(pass_true) + pass_false.var(ddof=1)/len(pass_false))
    confidence_interval = stats.t.interval(confidence_level, df=degrees_freedom, loc=mean_diff, scale=stderr)
    
    print(f"Mean (pass=True): {mean_pass_true}")
    print(f"Mean (pass=False): {mean_pass_false}")
    print(f"Difference in means: {mean_diff}")
    print(f'T-statistic: {t_stat}')
    print(f'P-value: {p_value}')
    print(f"95% confidence interval for the difference in means: {confidence_interval}")
    
    alpha = 0.05
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("There is no statistically significant difference between the groups.")
