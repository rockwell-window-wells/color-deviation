# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:58:43 2024

@author: Ryan.Larson
"""

import numpy as np
# import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76, rgb2lab
# from scipy.fft import fft2, ifft2, fftshift
import skimage.restoration as restoration
import plotly.graph_objects as go
import sys
from astropy.stats import sigma_clip

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

# def bgr_array_to_lab(bgr_array):
#     # Convert the BGR array to float32 and scale it to [0, 1]
#     bgr_array = np.float32(bgr_array) / 255.0
#     # Reshape to a single row image for conversion
#     bgr_image = bgr_array.reshape(1, -1, 3)
#     # Convert to Lab color space
#     lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
#     # Reshape back to the original array shape
#     lab_array = lab_image.reshape(-1, 3)
#     return lab_array

def bgr_array_to_lab(bgr_array):
    # Convert the BGR array to float32 and scale it to [0, 1]
    bgr_array = np.float32(bgr_array) / 255.0
    
    # Convert to Lab color space without reshaping
    lab_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2Lab)
    
    return lab_image

# def calculate_deltaE_lab_array(lab_array):
#     m = lab_array.shape[0]
#     delta_e_matrix = np.zeros((m, m))
#     for i in range(m):
#         for j in range(i, m):
#             delta_e = deltaE_cie76(lab_array[i], lab_array[j])
#             delta_e_matrix[i, j] = delta_e
#             delta_e_matrix[j, i] = delta_e  # Symmetric matrix
#     return delta_e_matrix

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

#     # Fourier Transform analysis
# def power_spectrum(image):
#     f_transform = fft2(image)
#     f_transform_shifted = fftshift(f_transform)
#     power_spec = np.abs(f_transform_shifted)**2
#     return power_spec


# Noise estimation (determine the measurable delta E)
def estimate_noise(bgr_image):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = rgb_image / 255.0
    noise_estimates = restoration.estimate_sigma(rgb_image, multichannel=True)
    return noise_estimates

def rgb_to_lab_single(rgb_color):
    """
    Convert a single RGB color to Lab color space.
    
    Parameters:
    rgb_color (list or tuple): A list or tuple of 3 float64 values representing an RGB color.
    
    Returns:
    list: A list of 3 float64 values representing the Lab color.
    """
    # Convert input to a NumPy array and normalize if needed
    rgb_color_np = np.array(rgb_color, dtype=np.float64)
    
    # Reshape to 1x1x3 to match the input shape expected by rgb2lab
    rgb_color_np = rgb_color_np.reshape((1, 1, 3))
    
    # Convert from RGB to Lab
    lab_color_np = rgb2lab(rgb_color_np)
    
    # Extract the Lab values from the resulting array
    lab_color = lab_color_np[0, 0, :]
    
    # Return as list
    return lab_color.tolist()

def calculate_max_neighbor_deviation(lab_image, neighbor_size=1):
    """
    Calculate the maximum delta E for each pixel based on its neighbors.
    
    Parameters:
        lab_image (np.ndarray): The input Lab image.
        neighbor_size (int): The half-width of the neighborhood to consider. For a 4x4 neighborhood, use 2.
        
    Returns:
        np.ndarray: Array of maximum delta E values for each pixel based on its neighbors.
    """
    h, w, _ = lab_image.shape
    max_delta_e = np.zeros((h, w), dtype=np.float32)
    
    print('\nCalculating max neighbor delta E...\n')
    try:
        for i in range(neighbor_size, h - neighbor_size):
            # Create the percentage complete string
            percent_complete = f"\rProgress: {100*(i+1)/(h - neighbor_size):.2f}%"
            
            # Use sys.stdout.write() to write to the terminal
            sys.stdout.write(percent_complete)
            
            # Flush the output buffer to ensure it prints immediately
            sys.stdout.flush()
            
            # print(f'{(i-neighbor_size)/(h - 2*neighbor_size):.2f}% Complete', flush=True)
            for j in range(neighbor_size, w - neighbor_size):
                center = lab_image[i, j]
                neighbors = lab_image[i-neighbor_size:i+neighbor_size+1, j-neighbor_size:j+neighbor_size+1].reshape(-1, 3)
    
                # Exclude the center pixel from the neighbor list
                mask = ~np.all(neighbors == center, axis=1)
                neighbors = neighbors[mask]
    
                if neighbors.size > 0:
                    delta_e = np.sqrt(np.sum((neighbors - center) ** 2, axis=1))
                    max_delta_e[i, j] = np.max(delta_e)
                else:
                    max_delta_e[i, j] = 0  # or np.nan if you prefer to mark these pixels differently
    except:
        print('Error encountered')
    print('\nMax neighbor deviation successfully calculated')

    return max_delta_e


def sigma_clipping_stack_rgb(images, sigma=3):
    """
    Perform sigma clipping stacking on a list of RGB images.

    Parameters:
    - images: list of 3D numpy arrays (RGB images).
    - sigma: number of standard deviations for clipping.

    Returns:
    - master_flat: 3D numpy array representing the master flat frame in RGB.
    """
    # Stack images into a 4D numpy array (number of images, height, width, channels)
    image_stack = np.stack(images, axis=0)
    
    # Initialize an empty array for the master flat frame
    master_flat = np.zeros(image_stack.shape[1:])

    # Process each channel (R, G, B) separately
    for channel in range(image_stack.shape[-1]):  # Assuming last axis is channels
        # Extract the current channel across all images
        channel_stack = image_stack[:, :, :, channel]
        
        # Apply sigma clipping on each pixel location
        clipped_channel = sigma_clip(channel_stack, sigma=sigma, axis=0)
        
        # Compute the mean of the clipped images
        master_flat[:, :, channel] = np.mean(clipped_channel, axis=0)
    
    return master_flat

# Once we have the standard deviation of the noise, we can estimate the likely
# Lab color space deviation caused by noise alone. That way if we see that
# pixel values are getting too far out of the expected range, we can decide
# whether that is due to image noise or a real measurement.


if __name__ == "__main__":
    # imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/test_image (7).jpg'
    # imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/single_light (1) ROI 1.jpg'
    imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/poor_light ROI 1.jpg'
    # image = cv2.imread(imfile)
    image = cv2.imread(imfile)
    
    # Estimate noise from the image (per channel, not grayscale)
    rgb_noise = estimate_noise(image)
    three_sigma_rgb_noise = [3*noise for noise in rgb_noise]
    lab_noise = rgb_to_lab_single(rgb_noise)
    three_sigma_lab_noise = rgb_to_lab_single(three_sigma_rgb_noise)
    
    deltaE = np.sqrt(np.sum([noise**2 for noise in lab_noise]))
    three_sigma_deltaE = np.sqrt(np.sum([noise**2 for noise in three_sigma_lab_noise]))
    
    print(f'deltaE = {deltaE}')
    print(f'3 sigma deltaE = {three_sigma_deltaE}')
    
    # Set reference color and compute delta E matrix
    reference_lab_color = (78.960, 2.728, 12.231)
    # reference_lab_color = (79.43, 5.33, 1.47)
    lab_array = bgr_array_to_lab(image)
    blurred_lab = cv2.GaussianBlur(lab_array, (101,101), 0)
    
    neighbor_size = 1
    neighbor_delta_e = calculate_max_neighbor_deviation(blurred_lab, neighbor_size=neighbor_size)
    
    plt.figure(dpi=300)
    plt.imshow(neighbor_delta_e, cmap='viridis')
    plt.colorbar()
    plt.title(f'Blurred neighbor-based delta E\nneighbor_size={neighbor_size}')
    
    delta_e_matrix = calculate_deltaE_lab_array(lab_array, reference_lab_color)
    
    # Impose a fake mask by directly setting values as nan
    delta_e_matrix_masked = delta_e_matrix.copy()
    delta_e_matrix_masked[500:,:] = np.nan
    
    # blurred_delta_e = cv2.GaussianBlur(delta_e_matrix, (101,101), 0)
    blurred_delta_e = cv2.GaussianBlur(delta_e_matrix_masked, (101,101), 0)
    
    # # Plot results (more blur = less extreme delta E)
    # plt.figure(dpi=300)
    # plt.imshow(delta_e_matrix, cmap='viridis')
    # plt.colorbar()
    # plt.title('No blur')
    # # plt.savefig(f'Blurred_(0x0).png')
    # # plt.close()
    
    # Test blurring levels to see when texture is no longer visible
    start_size = 5
    end_size = 101
    for num in range(start_size, end_size+1):
        if num % 8 == 1:
            blurred_delta_e = cv2.GaussianBlur(delta_e_matrix, (num, num), 0)
            # plt.figure(dpi=300)
            # plt.imshow(blurred_delta_e, cmap='viridis')
            # plt.colorbar()
            # plt.title(f'Blur with kernel size ({num},{num})')
            # plt.savefig(f'Blurred_({num}x{num}).png')
            # plt.close()
            
    dim = 1000
    small_delta_e_matrix = cv2.resize(delta_e_matrix, (dim, dim), interpolation=cv2.INTER_AREA)
    small_blurred_delta_e = cv2.resize(blurred_delta_e, (dim, dim), interpolation=cv2.INTER_AREA)
    small_neighbor_delta_e = cv2.resize(neighbor_delta_e, (dim, dim), interpolation=cv2.INTER_AREA)
    
    xno = np.arange(small_delta_e_matrix.shape[1])
    yno = np.arange(small_delta_e_matrix.shape[0])
    xno, yno = np.meshgrid(xno, yno)
    
    noblur_surface = go.Surface(x=xno, y=yno, z=small_delta_e_matrix, colorscale='plasma', name='No Blur')
    
    xblur = np.arange(small_neighbor_delta_e.shape[1])
    yblur = np.arange(small_neighbor_delta_e.shape[0])
    xblur, yblur = np.meshgrid(xblur, yblur)
    
    blur_surface = go.Surface(x=xblur, y=yblur, z=small_neighbor_delta_e, colorscale='plasma', name='Blur')
    
    # xblur = np.arange(small_blurred_delta_e.shape[1])
    # yblur = np.arange(small_blurred_delta_e.shape[0])
    # xblur, yblur = np.meshgrid(xblur, yblur)
    
    # blur_surface = go.Surface(x=xblur, y=yblur, z=small_blurred_delta_e, colorscale='plasma', name='Blur')
    
    # fig = go.Figure(data=noblur_surface)
    figno = go.Figure(data=noblur_surface)
    figblur = go.Figure(data=blur_surface)
    
    # Set the layout for the figure
    figno.update_layout(
        title="No blur",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Delta E'
        ),
        autosize=True
    )
    
    figblur.update_layout(
        title="Blur",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Delta E'
        ),
        autosize=True
    )
    
    # Show the figure
    figno.show(renderer='browser')
    figblur.show(renderer='browser')
    
    
        
    # # Compute power spectrum of the corrected image
    # power_spec = power_spectrum(image)
    
    # # Normalize the power spectrum
    # # Use percentile clipping to limit the range of values
    # pmin, pmax = np.percentile(power_spec, [1, 99])
    # power_spec_clipped = np.clip(power_spec, pmin, pmax)
    
    # plt.figure(dpi=300)
    # plt.imshow(np.log1p(power_spec_clipped), cmap='viridis')  # Visualize power spectrum
    # plt.colorbar()
    # plt.title('Power Spectrum')
    # plt.show()
    
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