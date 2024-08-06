# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:58:43 2024

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76
from scipy.fft import fft2, ifft2, fftshift

#%% Methods
def calculate_deltaE(lab_image, reference_color: tuple):
    """
    Calculate the delta E for all pixels in an image, as well as Lab deviations
    per channel.

    Parameters
    ----------
    lab_image : numpy float32 array (m x n x 3)
        A reference image in Lab color space.
    reference_color : tuple, length 3
        The Lab color space definition for a reference color, expected to be
        a color that exists within lab_image but not required.

    Returns
    -------
    delta_e : numpy float32 array (m x n x 1)
        Array of the delta E values across the whole lab_image, relative to
        reference_color.
    deviations : tuple, length 3, of numpy float32 arrays (m x n x 1)
        Array of the L, a, and b channel-wise deviations from reference_color.

    """
    l_ref, a_ref, b_ref = reference_color
    l_deviation = lab_image[..., 0] - l_ref
    a_deviation = lab_image[..., 1] - a_ref
    b_deviation = lab_image[..., 2] - b_ref
    delta_e = np.sqrt(l_deviation**2 + a_deviation**2 + b_deviation**2)
    deviations = (l_deviation, a_deviation, b_deviation)
    
    return delta_e, deviations

def bgr_to_lab(bgr_image):
    bgr_image = np.float32(bgr_image) / 255.0
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    return lab_image

def bgr_array_to_lab(bgr_array):
    # Convert the BGR array to float32 and scale it to [0, 1]
    bgr_array = np.float32(bgr_array) / 255.0
    # Reshape to a single row image for conversion
    bgr_image = bgr_array.reshape(1, -1, 3)
    # Convert to Lab color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    # Reshape back to the original array shape
    lab_array = lab_image.reshape(-1, 3)
    return lab_array

def calculate_deltaE_lab_array(lab_array):
    m = lab_array.shape[0]
    delta_e_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            delta_e = deltaE_cie76(lab_array[i], lab_array[j])
            delta_e_matrix[i, j] = delta_e
            delta_e_matrix[j, i] = delta_e  # Symmetric matrix
    return delta_e_matrix

def resize_image(image):
    '''
    Convenience function for resizing images to fit the monitor.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    resized_image : TYPE
        DESCRIPTION.

    '''
    # Get screen resolution (example resolution: 1920x1080)
    screen_width = 1920
    screen_height = 1080
    # Define a margin ratio (e.g., 0.9 means 10% margin)
    margin_ratio = 0.9
    # Adjust the available space by the margin ratio
    adjusted_screen_width = screen_width * margin_ratio
    adjusted_screen_height = screen_height * margin_ratio
    # Determine the scale factor
    scale_factor = min(adjusted_screen_width / image.shape[1], adjusted_screen_height / image.shape[0])
    # Resize the image to fit the screen with margin
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    
    return resized_image

def show_resized_image(image, title_str):
    cv2.imshow(title_str, resize_image(image))

def quantize_image(image, K=2):
    # Color quantization using k-means clustering
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 2 # Number of colors
    _, label, colors = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    colors = np.uint8(colors)
    quantized_image = colors[label.flatten()]
    quantized_image = quantized_image.reshape((image.shape))
    
    return quantized_image, colors

def quantize_image_in_masked_area(image, mask, K=2):
    # Apply mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Extract non-black pixels
    non_black_pixels = masked_image[mask != 0].reshape(-1, 3)
    non_black_pixels = np.float32(non_black_pixels)

    # Perform k-means clustering on non-black pixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, colors = cv2.kmeans(non_black_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    colors = np.uint8(colors)

    # Create quantized image
    quantized_image = image.copy()
    quantized_image[mask != 0] = colors[label.flatten()]

    return quantized_image, colors

    # Fourier Transform analysis
def power_spectrum(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    power_spec = np.abs(f_transform_shifted)**2
    return power_spec


if __name__ == "__main__":
    imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/single_light (1) ROI 1.jpg'
    # imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/double_light (1) ROI 1.jpg'
    # image = cv2.imread(imfile)
    image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)
        
    # Compute power spectrum of the corrected image
    power_spec = power_spectrum(image)
    
    # Normalize the power spectrum
    # Use percentile clipping to limit the range of values
    pmin, pmax = np.percentile(power_spec, [1, 99])
    power_spec_clipped = np.clip(power_spec, pmin, pmax)
    
    plt.figure(dpi=300)
    plt.imshow(np.log1p(power_spec_clipped), cmap='viridis')  # Visualize power spectrum
    plt.colorbar()
    plt.title('Power Spectrum')
    plt.show()
    
    # # Apply a Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(image, (7, 7), 0)
    # K = 3
    # quantized_blurred, blurred_colors = quantize_image(blurred, K)
    # quantized_image, colors = quantize_image(image, K)
    
    # quantized_blurred_lab = bgr_to_lab(quantized_blurred)
    # reference_color = (75.075, -1.525, 10.267)
    # delta_e, deviations = calculate_deltaE(quantized_blurred_lab, reference_color)
    
    # # show_resized_image(quantized_blurred, f'Blurred Image: {K} Colors')
    # # show_resized_image(quantized_image, f'Quantized Image: {K} Colors')
    
    # # Mask any color values that are too dark
    # n_darkest = 2
    # brightness = np.sum(blurred_colors, axis=1)
    
    # sorted_indices = np.argsort(brightness)
    # darkest_indices = sorted_indices[:n_darkest]
    
    # darkest_colors = blurred_colors[darkest_indices]
    
    # mask = np.zeros((quantized_blurred.shape[0], quantized_blurred.shape[1]), dtype=bool)
    
    # # Check each pixel in the quantized image
    # for color in darkest_colors:
    #     mask |= np.all(quantized_blurred == color, axis=-1)
        
    # mask = np.expand_dims(mask, axis=-1)
    
    # # darkthresh = 100
    # # mask = quantized_blurred <= darkthresh
    
    # # Apply mask to blurred
    # blurred_masked = blurred * ~mask
    # # blurred_masked = np.where(~mask, np.nan, blurred)
    # # show_resized_image(blurred_masked, "Blurred Masked")
    
    # # Convert to grayscale for binary operations
    # gray_quantized = cv2.cvtColor(blurred_masked, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray_quantized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # # Define a kernel for morphological operations
    # kernel = np.ones((7, 7), np.uint8)

    # # Dilation and Erosion
    # iterations = 30
    # overshoot = 25
    # binary = cv2.erode(binary, kernel, iterations=iterations)
    # binary = cv2.dilate(binary, kernel, iterations=iterations+overshoot)
    # binary = cv2.erode(binary, kernel, iterations=overshoot)
    # dilated_masked = cv2.bitwise_and(blurred, blurred, mask=binary)
    # # show_resized_image(dilated_masked, "Dilated Mask")
    
    
    # # Quantize the masked image
    # K = 4
    # quantized_masked, masked_colors = quantize_image_in_masked_area(dilated_masked, binary, K=K)
    # show_resized_image(quantized_masked, "Quantized Masked")
    
    
    # # for _ in range(iterations):
    # #     binary = cv2.dilate(binary, kernel, iterations=1)   # Dilate
    
    # # for _ in range(iterations):
    # #     binary = cv2.erode(binary, kernel, iterations=1)    # Erode

    # # # Opening (Erosion followed by Dilation)
    # # opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # # # Closing (Dilation followed by Erosion)
    # # iterations = 100
    # # for _ in range(iterations):
    # #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # # dilated_masked = cv2.bitwise_and(blurred, blurred, mask=binary)
    # # show_resized_image(dilated_masked, "Dilated Mask")
    
    # # # Quantize blurred_masked
    # # K = 3
    # # quantized_blurred_masked
    
    # # cmap = plt.get_cmap('jet')
    # # fig, ax = plt.subplots(dpi=300)
    # # cax = ax.imshow(delta_e, cmap=cmap, interpolation='bilinear')
    # # cbar = plt.colorbar(cax, ax=ax)
    # # cbar.set_label('Delta E')

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # Calculate delta E between each quantized color in the end (one of the
    # # colors will be black, so this will always have some outliers)
    # lab_array = bgr_array_to_lab(masked_colors)
    # delta_e_matrix = calculate_deltaE_lab_array(lab_array)