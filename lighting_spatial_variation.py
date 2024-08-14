# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:23:34 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import color

# Create the initial RGB image with sinusoidal variation
width, height = 2000, 2000
base_color = np.array([237, 235, 235])

# Generate the sine wave variation
x = np.arange(width)
y = np.arange(height)
xv, yv = np.meshgrid(x, y)

# Sinusoidal variation with a period of 100 pixels
period = 200
variation = 5 * (np.sin(2 * np.pi * xv / period) + np.sin(2 * np.pi * yv / period))

# Add the variation to the base color
image_rgb = np.clip(base_color + variation[:, :, np.newaxis], 0, 255).astype(np.uint8)

# Convert the image to Lab color space
image_lab = color.rgb2lab(image_rgb)

# Define the sinusoidal darkening gradient
L_value = 5  # Level of darkening
x_center = width // 4

# Create a sinusoidal darkening effect that affects the L channel
darkening = L_value * np.sin(np.pi * (xv - x_center) / (width // 2))
shift = 80
darkening_shifted = L_value * np.sin(np.pi * (xv - x_center - shift) / (width // 2))
# darkening = darkening[:, np.newaxis]  # Add a new axis for broadcasting

# Apply the darkening effect to the L channel
image_lab[:, :, 0] = np.clip(image_lab[:, :, 0] + darkening, 0, 100)

# Convert back to RGB
image_rgb_darkened = np.clip(color.lab2rgb(image_lab) * 255, 0, 255).astype(np.uint8)

# Convert darkened image to Lab
image_lab_darkened = color.rgb2lab(image_rgb_darkened)

image_lab_restored = image_lab_darkened.copy()
image_lab_restored[:,:,0] = image_lab_restored[:,:,0] - darkening_shifted

image_rgb_restored = np.clip(color.lab2rgb(image_lab_restored) * 255, 0, 255).astype(np.uint8)

# Display the resulting images
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# plt.axis('off')
# plt.title(f'Original image')

# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb_darkened)
# plt.axis('off')
# plt.title(f'Darkened image')

# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb_restored)
# plt.axis('off')
# plt.title(f'Restored image')

plt.figure(figsize=(10, 10), dpi=500)
plt.imshow(image_rgb_restored-image_rgb, cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.title(f'Difference between restored and original images\nShift: {shift}')

plt.show()

# # Save the resulting image
# cv2.imwrite("darkened_image.png", cv2.cvtColor(image_rgb_darkened, cv2.COLOR_RGB2BGR))
