# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:19:50 2024

@author: Ryan.Larson
"""

import numpy as np
from astropy.stats import sigma_clip
from picamera2 import Picamera2, Preview
from time import sleep
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def capture_images(directory, num_images=10):
    """
    Capture a set number of images using Picamera2 and save them to a directory.

    Parameters:
    - directory: Directory where the images will be saved.
    - num_images: Number of images to capture.
    """
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start_preview(Preview.QTGL)
    sleep(2)  # Allow the camera to adjust

    for i in range(num_images):
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
    images = []
    for image_path in sorted(Path(directory).glob("*.png")):
        img = Image.open(image_path)
        images.append(np.array(img))
    return images


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


def save_and_display_flat_field(master_flat, output_file):
    """
    Save the flat field image to a file and display it.

    Parameters:
    - master_flat: 3D numpy array representing the master flat frame in RGB.
    - output_file: Path where the flat field image will be saved.
    """
    # Convert to Image object and save in raw format
    flat_image = Image.fromarray(np.uint8(master_flat))
    flat_image.save(output_file)
    
    # Display the flat field image
    plt.imshow(master_flat.astype(np.uint8))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Define the directory to save images and the number of images to capture
    image_directory = "/home/pi/flat_field_images"
    num_images_to_capture = 10

    # Create the directory if it doesn't exist
    Path(image_directory).mkdir(parents=True, exist_ok=True)

    # Capture images using Picamera2
    capture_images(image_directory, num_images=num_images_to_capture)

    # Load images from the directory
    images = load_images_from_directory(image_directory)

    # Perform sigma clipping stacking
    master_flat = sigma_clipping_stack_rgb(images, sigma=3)

    # Save and display the flat field image
    output_path = "/home/pi/flat_field.dng"  # Output file path
    save_and_display_flat_field(master_flat, output_path)