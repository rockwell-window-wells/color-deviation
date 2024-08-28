# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:24:17 2024

@author: Ryan.Larson
"""

import os
import csv
import pandas as pd
import cv2
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np
from skimage.color import deltaE_cie76

# def rgb_to_lab(rgb):
#     rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2])
#     lab_color = convert_color(rgb_color, LabColor)
#     return lab_color.lab_l, lab_color.lab_a, lab_color.lab_b

# # Vectorize the function
# vectorized_rgb_to_lab = np.vectorize(rgb_to_lab, signature='(n)->(m)')

# # Now, if you want to convert back to RGB
# def lab_to_rgb(lab):
#     lab_color = LabColor(lab[0], lab[1], lab[2])
#     rgb_color = convert_color(lab_color, sRGBColor)
#     return rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b

# # Vectorize the function
# vectorized_lab_to_rgb = np.vectorize(lab_to_rgb, signature='(n)->(m)')

# def compute_delta_e_vectorized(lab_array, reference_lab):
#     delta_e_values = np.zeros(lab_array.shape[0])
    
#     for i, lab in enumerate(lab_array):
#         delta_e_values[i] = delta_e_cie1976(LabColor(*lab), reference_lab)
    
#     return delta_e_values

def calculate_deltaE_lab_array(lab_array, reference_lab_color):
    # lab_array should have the shape (height, width, 3)
    height, width, _ = lab_array.shape
    
    # Reshape lab_array to (height*width, 3) for vectorized computation
    reshaped_lab = lab_array.reshape(-1, 3)
    
    # Create an array for the reference color with the same shape as reshaped_lab
    reference_lab_array = np.tile(reference_lab_color, (reshaped_lab.shape[0], 1))
    
    # Calculate the Delta E values for each pixel
    delta_e_array = deltaE_cie76(reshaped_lab, reference_lab_array)
    
    # Reshape the delta_e_array back to the original image's height and width
    delta_e_matrix = delta_e_array.reshape(height, width)
    return delta_e_matrix

def process_file(file_path):
    # Placeholder for your custom processing logic
    # For example, you could read the file, perform calculations, etc.
    # Return some data to be written to the CSV
    # with open(file_path, 'r') as file:
    #     data = file.read()
    img = cv2.imread(file_path)
    img_float = img.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    # img_lab = np.empty(img.shape)
    # print(f'Processing {file_path}')
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         b = img[i][j][0] / 255
    #         g = img[i][j][1] / 255
    #         r = img[i][j][2] / 255
    #         rgb = (r, g, b)
    #         img_lab[i][j][0], img_lab[i][j][1], img_lab[i][j][2], = rgb_to_lab(rgb)
            
    # lab_array = vectorized_rgb_to_lab(img)
    ref_lab = (79.69, 1.75, 5.75)
    delta_e_array = calculate_deltaE_lab_array(img_lab, ref_lab)
    print(f'{os.path.basename(file_path)}\t{np.min(delta_e_array)}')
    return (np.min(delta_e_array), np.max(delta_e_array), np.median(delta_e_array))

def process_subdirectory(subdirectory_path):
    # Initialize a list to store processed data
    processed_data = []

    # Iterate over all files in the subdirectory
    for filename in os.listdir(subdirectory_path):
        file_path = os.path.join(subdirectory_path, filename)

        # Ensure it's a file before processing
        if os.path.isfile(file_path) and filename.lower().endswith('.jpg'):
            min_deltaE, max_deltaE, median_deltaE = process_file(file_path)
            processed_data.append([filename, min_deltaE, max_deltaE, median_deltaE])

    # Define the output CSV file path
    csv_output_path = os.path.join(subdirectory_path, f'output_{os.path.split(subdirectory_path)[-1]}.csv')

    # Write the processed data to the CSV file
    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Min Delta E', 'Max Delta E', 'Median Delta E'])  # CSV headers
        csv_writer.writerows(processed_data)

def process_main_directory(main_directory):
    # Iterate over each subdirectory in the main directory
    for subdirectory_name in os.listdir(main_directory):
        subdirectory_path = os.path.join(main_directory, subdirectory_name)

        # Ensure it's a directory before processing
        if os.path.isdir(subdirectory_path):
            process_subdirectory(subdirectory_path)

if __name__ == "__main__":
    main_directory = "annotated_images"
    process_main_directory(main_directory)