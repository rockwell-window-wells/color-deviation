# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:58:43 2024

@author: Ryan.Larson
"""

import numpy as np
# import pandas as pd
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.color import deltaE_cie76, rgb2lab
# from scipy.fft import fft2, ifft2, fftshift
import skimage.restoration as restoration
# import plotly.graph_objects as go
import sys
from astropy.stats import sigma_clip
from picamera2 import Picamera2
from time import sleep
from pathlib import Path
from PIL import Image

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
    
    # Convert to Lab color space without reshaping
    lab_image = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2Lab)
    return lab_image

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
    screen_width = 800
    screen_height = 480
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

def capture_images(directory, num_images=10):
    """
    Capture a set number of images using Picamera2 and save them to a directory.

    Parameters:
    - directory: Directory where the images will be saved.
    - num_images: Number of images to capture.
    """
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    # picam2.start_preview(QtPreview())
    picam2.start()
    sleep(2)  # Allow the camera to adjust

    for i in range(num_images):
        print(f'Capturing image_{i:03d}.png')
        image_path = Path(directory) / f"image_{i:03d}.png"
        picam2.capture_file(str(image_path))
        sleep(1)  # Small delay between captures

    picam2.close()

def load_images_from_directory(directory):
    """
    Load all images from a directory into a list of 3D numpy arrays.

    Parameters:
    - directory: Directory containing the images.

    Returns:
    - images: List of 3D numpy arrays (RGB images).
    """
    print('Loading images')
    images = []
    for image_path in sorted(Path(directory).glob("*.png")):
        img = Image.open(image_path)
        images.append(np.array(img))
    return images

def flat_field_correct(target_image, flat_field):
    # Ensure that the target image and the flat field image are the same size
    assert target_image.shape == flat_field.shape
    
    target_image = target_image.astype(np.float32)
    flat_field = flat_field.astype(np.float32)
    
    flat_corrected = flat_field / np.median(flat_field)
    
    calibrated_image = target_image / flat_corrected
    
    return calibrated_image
    

# Once we have the standard deviation of the noise, we can estimate the likely
# Lab color space deviation caused by noise alone. That way if we see that
# pixel values are getting too far out of the expected range, we can decide
# whether that is due to image noise or a real measurement.


if __name__ == "__main__":
    # directory = "test_images"
    
    # picam2 = Picamera2()
    # config = picam2.create_still_configuration(
        # main={"size": (4056, 3040)},
    # )
    # picam2.configure(config)
    
    # picam2.set_controls({
        # "ExposureTime": 20000,
        # "AnalogueGain": 1.0,
        # "AwbEnable": True,
        # "ColourGains": (1.0, 1.0, 1.0)
    # })
    
    # picam2.start()
    # image = picam2.capture_array()
    # picam2.stop()
    
    # print('[INFO] Picture taken')
    
    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('test_histogram.png', image_bgr)
    
    # plt.figure(figsize=(4,4), dpi=100)
    # plt.imshow(image)
    # plt.colorbar()
    # plt.title('Test')
    # plt.show()
    
    # Set reference color and compute delta E matrix
    reference_lab_color = (79.69, 1.75, 5.75)
    
    directory = Path('/home/rweng/Documents/GitHub/color-deviation/test_images')
    
    min_deltaE = []
    max_deltaE = []
    median_deltaE = []
    
    files = sorted([file for file in directory.glob('*.png') if file.is_file()])
    
    for file in files:
        if '_deltaE' in file.stem:
            continue
        
        image = cv2.imread(file)
        
        # Correct the image for flat field
        flat_field = cv2.imread("/home/rweng/Documents/GitHub/color-deviation/flat_field.png")
        calibrated_image = flat_field_correct(image, flat_field)
        lab_array = bgr_array_to_lab(calibrated_image)
        lab_array = bgr_array_to_lab(image)
        
        delta_e_matrix = calculate_deltaE_lab_array(lab_array, reference_lab_color)
        
        min_deltaE.append(np.min(delta_e_matrix))
        max_deltaE.append(np.max(delta_e_matrix))
        median_deltaE.append(np.median(delta_e_matrix))
        
        print(f'\n{file.name}:')
        print(f'Min:\t{np.min(delta_e_matrix):.2f}')
        print(f'Max:\t{np.max(delta_e_matrix):.2f}')
        print(f'Median:\t{np.median(delta_e_matrix):.2f}')
        
        # Plot results (more blur = less extreme delta E)
        # plt.figure(figsize=(4,4), dpi=200)
        # plt.imshow(delta_e_matrix, cmap='viridis')
        # plt.colorbar()
        # plt.title('Base image delta E')
        
        # file_stem = file.stem
        # file_extension = file.suffix
        
        # plot_filename = f"{file_stem}_deltaE{file_extension}"
        
        # plot_filepath = file.with_name(plot_filename)
        
        # plt.savefig(plot_filepath, format='png')
        
        # plt.close()
        
        # plt.show()
        
    print('#'*60)
    print(f'Median Min:\t{np.median(min_deltaE):.2f}')
    print(f'Std Dev Min:\t{np.std(min_deltaE):.2f}')
    print(f'99.7% Conf Min:\t({np.median(min_deltaE)-3*np.std(min_deltaE):.2f}, {np.median(min_deltaE)+3*np.std(min_deltaE):.2f})')
    
    print(f'\nMedian Max:\t{np.median(max_deltaE):.2f}')
    print(f'Std Dev Max:\t{np.std(max_deltaE):.2f}')
    print(f'99.7% Conf Max:\t({np.median(max_deltaE)-3*np.std(max_deltaE):.2f}, {np.median(max_deltaE)+3*np.std(max_deltaE):.2f})')
    
    print(f'\nMedian Median:\t{np.median(median_deltaE):.2f}')
    print(f'Std Dev Median:\t{np.std(median_deltaE):.2f}')
    print(f'99.7% Conf Median:\t({np.median(median_deltaE)-3*np.std(median_deltaE):.2f}, {np.median(median_deltaE)+3*np.std(median_deltaE):.2f})')