# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 07:42:07 2024

@author: Ryan.Larson
"""

from gimpfu import *
import math

def find_local_mins(histvals):
    min_indices = []
    for i in range(len(histvals)):
        if i == 0:
            if histvals[i] < histvals[i + 1]:
                min_indices.append(i)
        elif i == len(histvals) - 1:
            if histvals[i] < histvals[i - 1]:
                min_indices.append(i)
        else:
            if histvals[i] < histvals[i - 1] and histvals[i] < histvals[i + 1]:
                min_indices.append(i)
    
    return min_indices

def find_local_maxs(histvals):
    max_indices = []
    for i in range(len(histvals)):
        if i == 0:
            if histvals[i] > histvals[i + 1]:
                max_indices.append(i)
        elif i == len(histvals) - 1:
            if histvals[i] > histvals[i - 1]:
                max_indices.append(i)
        else:
            if histvals[i] > histvals[i - 1] and histvals[i] > histvals[i + 1]:
                max_indices.append(i)
    
    return max_indices

def gaussian_kernel(size, sigma=1):
    """Creates a 1D Gaussian kernel."""
    kernel = [math.exp(-(x - size//2)**2 / (2 * sigma**2)) for x in range(size)]
    # Normalize the kernel so that the sum is 1
    kernel_sum = sum(kernel)
    return [x / kernel_sum for x in kernel]

def gaussian_smooth(histogram, kernel_size=5, sigma=1):
    """Applies Gaussian smoothing to a histogram."""
    kernel = gaussian_kernel(kernel_size, sigma)
    half_size = kernel_size // 2
    smoothed_histogram = []
    
    for i in range(len(histogram)):
        weighted_sum = 0.0
        for j in range(kernel_size):
            # Calculate the index for the histogram value
            index = i + j - half_size
            # Handle boundaries by reflecting the index
            if index < 0:
                index = -index
            elif index >= len(histogram):
                index = 2*len(histogram) - index - 2
            # Apply the kernel
            weighted_sum += histogram[index] * kernel[j]
        
        smoothed_histogram.append(weighted_sum)
    
    return smoothed_histogram

def find_black_point(histvals):
    """Uses Gaussian smoothing and local max and min values to determine the
    number of large structures in the data, and then where the black point
    should be raised to."""
    
    # Smooth the data
    kernel_size = 31
    sigma = 10
    smoothed_histvals = gaussian_smooth(histvals, kernel_size=kernel_size, sigma=sigma)
    
    # Find local min and max indices
    min_indices = find_local_mins(smoothed_histvals)
    max_indices = find_local_maxs(smoothed_histvals)
    
    # Create a sorted list of indices with corresponding Booleans to indicate
    # a maximum (True for max, False for min)
    local_extrema_indices = []
    local_extrema = []
    local_max = []
    imin = 0
    imax = 0
    while (imin + imax) < (len(min_indices) + len(max_indices)):
        if imin == len(min_indices):
            local_extrema_indices.append(max_indices[imax])
            local_max.append(True)
            local_extrema.append(smoothed_histvals[max_indices[imax]])
            imax += 1
            continue
        if imax == len(max_indices):
            local_extrema_indices.append(min_indices[imin])
            local_max.append(False)
            local_extrema.append(smoothed_histvals[min_indices[imin]])
            imin += 1
            continue
        if min_indices[imin] < max_indices[imax]:
            local_extrema_indices.append(min_indices[imin])
            local_max.append(False)
            local_extrema.append(smoothed_histvals[min_indices[imin]])
            imin += 1
        else:
            local_extrema_indices.append(max_indices[imax])
            local_max.append(True)
            local_extrema.append(smoothed_histvals[max_indices[imax]])
            imax += 1
    
    # Determine which minima are the lowest in proportion by forward division
    forward_percentages = []
    for i in range(len(local_extrema)):
        if i == len(local_extrema) - 1:
            break
        
        pct = local_extrema[i] / local_extrema[i+1]
        forward_percentages.append(pct)
    
    pct_thresh = 0.1
    
    for i, pct in enumerate(forward_percentages):
        if pct < pct_thresh:
            black_point_guess = local_extrema_indices[i]
            break
    
    if black_point_guess < 90:
        black_point_guess = 90
    
    # Adjust the black point if necessary
    while histvals[black_point_guess] < 1000:
        black_point_guess += 1
        
    black_point = black_point_guess
    
    return black_point

# def find_black_point(histvals):
#     # Inform the guess based on the number of structures in the histogram
#     structures = detect_large_structures(histvals, threshold_percentage=0.01)
#     nstructures = len(structures)
    
#     # If there are 2 structures, there is a typical range to look for the 
#     # new black point
#     if nstructures > 1:
#         # print "More than 1 structure found"
#         min_black = 130
#         max_black = 160
#     else:
#         min_black = 109
#         max_black = 255
        
#     min_indices = find_local_mins(histvals) # Get the indices of all local minima
#     min_indices_filtered = [i for i in min_indices if i >= min_black and i <= max_black]
#     minvals = [histvals[i] for i in min_indices_filtered]
#     if not minvals:
#         return 0  # If no minima are found, return a default value
#     min_minvals_indices = find_local_mins(minvals)
#     minval = minvals[min_minvals_indices[0]]
    
#     black_point_guess = histvals.index(minval)
    
#     black_thresh = 0.01
    
#     if black_point_guess == 0:
#         for i, val in enumerate(histvals):
#             if i > min_black:
#                 if val >= black_thresh:
#                     black_point = i
#                     break
#     else:
#         black_point = black_point_guess  # Set black_point to the guessed value

#     return black_point

# def find_peaks(data):
#     peaks = []
#     for i in range(1, len(data) - 1):
#         if data[i-1] < data[i] > data[i+1]:
#             peaks.append((i, data[i]))
#     return peaks

# def detect_large_structures(data, threshold_percentage=0.2):
#     peaks = find_peaks(data)
    
#     if not peaks:
#         return 0  # No peaks found
    
#     # Get the height of the maximum peak
#     max_peak_value = max(peaks, key=lambda x: x[1])[1]
#     threshold_value = max_peak_value * threshold_percentage
    
#     structures = []
#     inside_structure = False
#     start_index = None

#     for i, value in enumerate(data):
#         if value >= threshold_value:
#             if not inside_structure:
#                 inside_structure = True
#                 start_index = i  # Mark the start of a new structure
#         else:
#             if inside_structure:
#                 inside_structure = False
#                 structures.append((start_index, i))  # Mark the end of the structure

#     # If still inside a structure at the end of data
#     if inside_structure:
#         structures.append((start_index, len(data)))

#     # Count how many significant structures were found
#     return structures

def get_histogram_from_layer(layer):
    num_bins = 256
    histogram = [pdb.gimp_histogram(layer, HISTOGRAM_VALUE, i, i) for i in range(num_bins)]
    normalized_hist = [row[5] for row in histogram]
    histvals = [row[4] for row in histogram]
    return normalized_hist, histvals

def add_reference_mask(image, filename):
    pdb.gimp_image_undo_group_start(image)
    layer = pdb.gimp_file_load_layer(image, filename)
    pdb.gimp_image_add_layer(image, layer, 0)
    pdb.gimp_image_undo_group_end(image)
    
def invert_and_subtract_mask(image):
    pdb.gimp_drawable_invert(image.active_layer, 0)
    pdb.gimp_layer_set_mode(image.active_layer, SUBTRACT_MODE)
    pdb.gimp_image_merge_down(image, image.active_layer, 2)

def process_7038step1(image, drawable):
    # Use the provided image and drawable instead of getting all open images
    # pdb.gimp_message(f"Processing image: {image.name}")  # Debugging message
    
    layer = drawable  # Use the provided drawable as the layer
    normalized_hist, histvals = get_histogram_from_layer(layer)
    black_point = find_black_point(histvals)
    # black_point = find_black_point(normalized_hist)
    pdb.gimp_levels(layer, HISTOGRAM_VALUE, black_point, 255, 1.0, 0, 255)
    
    # Check if the image is grayscale
    if pdb.gimp_image_base_type(image) == RGB:
        # Convert to grayscale if the image is RGB
        pdb.gimp_image_convert_grayscale(image)
    
    new_layer = pdb.gimp_layer_copy(layer, False)
    # pdb.gimp_image_add_layer(image, new_layer, 0)
    pdb.gimp_image_insert_layer(image, new_layer, None, 0)
    new_layer.name = "Thermal Shock"
    
    gimp.displays_flush()

def register_script():
    register(
        "7038_step1",
        "Thermal shock 7038 step 1",
        "Sets black point automatically and changes image mode to grayscale",
        "Ryan Larson",
        "GNU GPLv3",
        "2024",
        "<Image>/Filters/Custom/7038 Step 1",
        "RGB*",  # Specify acceptable image types
        [],
        [],
        process_7038step1)

register_script()
main()
