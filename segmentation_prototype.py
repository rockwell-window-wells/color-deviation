# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:53:38 2024

@author: Ryan.Larson
"""

import cv2
import numpy as np

def resize_to_screen(image, max_width=800, max_height=600):
    """Resize the image to fit within the max_width and max_height while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        # Calculate the scaling factor and resize
        scaling_factor = min(max_width / float(w), max_height / float(h))
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

# Load image
image = cv2.imread('C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/lighting_test (3).jpg')
# image = cv2.imread('C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/test_image (1).jpg')
# image = cv2.imread('C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png')
image_resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # Resize for faster processing

# Color quantization using k-means clustering
Z = image_resized.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2 # Number of colors
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
quantized_image = center[label.flatten()]
quantized_image = quantized_image.reshape((image_resized.shape))

# Show the quantized image
cv2.imshow('Quantized Image', resize_to_screen(quantized_image))

# Convert to grayscale for binary operations
gray_quantized = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray_quantized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Dilation and Erosion
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

masked_image = cv2.bitwise_and(quantized_image, quantized_image, mask=closing)

# Display the results
# cv2.imshow('Quantized Image', resize_to_screen(quantized_image))
# cv2.imshow('Binary Image', resize_to_screen(binary))
# cv2.imshow('Dilated Image', resize_to_screen(dilated))
# cv2.imshow('Eroded Image', resize_to_screen(eroded))
# cv2.imshow('Opening Image', resize_to_screen(opening))
# cv2.imshow('Closing Image', resize_to_screen(closing))
# cv2.imshow('Masked Image', resize_to_screen(masked_image))

cv2.waitKey(0)
cv2.destroyAllWindows()

# # Convert to grayscale for further processing
# gray = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2GRAY)

# # Preprocessing
# blurred = cv2.GaussianBlur(gray, (15, 15), 0)
# cv2.imshow('Blurred Image', resize_to_screen(blurred))

# # Edge detection
# edges = cv2.Canny(blurred, 30, 100)
# cv2.imshow('Edges', resize_to_screen(edges))

# # Morphological operations
# kernel = np.ones((5, 5), np.uint8)
# dilated = cv2.dilate(edges, kernel, iterations=1)
# closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Dilated Edges', resize_to_screen(dilated))
# cv2.imshow('Closed Image', resize_to_screen(closed))

# # Contour detection and filling
# contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# masks = np.zeros_like(gray)
# for contour in contours:
#     if cv2.contourArea(contour) > 500:  # Filter small areas
#         cv2.drawContours(masks, [contour], -1, (255), thickness=-1)

# # Show final mask
# cv2.imshow('Masks', resize_to_screen(masks))

# # Post-processing and mask creation
# num_labels, labels = cv2.connectedComponents(masks)
# for i in range(1, num_labels):  # Ignore background
#     mask = (labels == i).astype(np.uint8) * 255
#     cv2.imshow(f'Mask {i}', resize_to_screen(mask))
#     cv2.imwrite(f'mask_{i}.png', mask)

# cv2.waitKey(0)
# cv2.destroyAllWindows()





# def resize_to_screen(image, max_width=800, max_height=600):
#     """Resize the image to fit within the max_width and max_height while maintaining aspect ratio."""
#     h, w = image.shape[:2]
#     if h > max_height or w > max_width:
#         # Calculate the scaling factor and resize
#         scaling_factor = min(max_width / float(w), max_height / float(h))
#         image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#     return image

# # Load image
# image = cv2.imread('C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Show original and grayscale images
# cv2.imshow('Original Image', resize_to_screen(image))
# # cv2.imshow('Grayscale Image', resize_to_screen(gray))

# # # Preprocessing
# # blurred = cv2.GaussianBlur(gray, (51, 51), 0)
# # edges = cv2.Canny(blurred, 50, 150)
# # cv2.imshow('Blurred Image', resize_to_screen(blurred))
# # cv2.imshow('Edges', resize_to_screen(edges))

# # Histogram equalization
# equalized = cv2.equalizeHist(gray)
# blurred = cv2.GaussianBlur(equalized, (51, 51), 0)
# cv2.imshow('Equalized Image', resize_to_screen(equalized))
# cv2.imshow('Blurred Image', resize_to_screen(blurred))
# edges = cv2.Canny(blurred, 50, 150)

# # Morphological operations
# kernel = np.ones((5, 5), np.uint8)
# dilated = cv2.dilate(edges, kernel, iterations=1)
# closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Dilated Edges', resize_to_screen(dilated))
# cv2.imshow('Closed Image', resize_to_screen(closed))

# # Contour detection and filling
# contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# masks = np.zeros_like(gray)
# for contour in contours:
#     if cv2.contourArea(contour) > 500:  # Filter small areas
#         cv2.drawContours(masks, [contour], -1, (255), thickness=-1)

# # Show final mask
# cv2.imshow('Masks', resize_to_screen(masks))

# # Post-processing and mask creation
# num_labels, labels = cv2.connectedComponents(masks)
# for i in range(1, num_labels):  # Ignore background
#     mask = (labels == i).astype(np.uint8) * 255
#     # cv2.imshow(f'Mask {i}', resize_to_screen(mask))
#     # cv2.imwrite(f'mask_{i}.png', mask)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
