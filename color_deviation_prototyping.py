# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:43:11 2024

@author: Ryan.Larson
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """Calculate the color histogram of an image in LAB color space."""
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Compute histogram for each channel
    l_hist = cv2.calcHist([lab_image[..., 0]], [0], None, [256], [0, 256])
    a_hist = cv2.calcHist([lab_image[..., 1]], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([lab_image[..., 2]], [0], None, [256], [0, 256])

    return l_hist.flatten(), a_hist.flatten(), b_hist.flatten()

def calculate_deviation(lab_image, reference_color):
    """Calculate the minimum and maximum deviation from the reference LAB color."""
    l_ref, a_ref, b_ref = reference_color

    # Calculate the deviations
    l_deviation = np.abs(lab_image[..., 0] - l_ref)
    a_deviation = np.abs(lab_image[..., 1] - a_ref)
    b_deviation = np.abs(lab_image[..., 2] - b_ref)

    # Compute minimum and maximum deviations
    min_deviation = np.min([l_deviation, a_deviation, b_deviation])
    max_deviation = np.max([l_deviation, a_deviation, b_deviation])
    # min_deviation = np.min([l_deviation, a_deviation, b_deviation], axis=0)
    # max_deviation = np.max([l_deviation, a_deviation, b_deviation], axis=0)

    return min_deviation, max_deviation, np.max(l_deviation), np.max(a_deviation), np.max(b_deviation)

def main(image_path, reference_lab):
    """Main function to process the image and compute deviations."""
    # Read the image
    image = cv2.imread(image_path)

    # Calculate color histogram
    l_hist, a_hist, b_hist = calculate_histogram(image)
    # print("L channel histogram:", l_hist.flatten())
    # print("A channel histogram:", a_hist.flatten())
    # print("B channel histogram:", b_hist.flatten())
    fig = plt.figure(dpi=300)
    plt.plot(l_hist, label="L")
    plt.plot(a_hist, label="a")
    plt.plot(b_hist, label="b")
    plt.legend()
    plt.show()

    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Calculate deviations
    min_deviation, max_deviation, l_deviation, a_deviation, b_deviation = calculate_deviation(lab_image, reference_lab)
    print("Minimum deviation from reference color:", min_deviation)
    print("Maximum deviation from reference color:", max_deviation)
    print("Maximum L deviation from reference color:", l_deviation)
    print("Maximum a deviation from reference color:", a_deviation)
    print("Maximum b deviation from reference color:", b_deviation)
    

if __name__ == "__main__":
    # Example usage
    image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png'  # Update with your image path
    reference_lab = (79.43, 5.33, 1.47)  # Update with your LAB reference color

    main(image_path, reference_lab)
