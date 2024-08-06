# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:46:36 2024

@author: Ryan.Larson
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def power_spectrum(image):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    power_spec = np.abs(f_transform_shifted)**2
    return power_spec

# Load the image (grayscale)
imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/single_light (1) ROI 1.jpg'
image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

# Compute the Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Create a mask to remove the frequency near y=1500
rows, cols = image.shape
mask = np.ones((rows, cols), dtype=np.uint8)
y_freq_to_remove = 1600

# Define a band around the target frequency to remove
band_width = 10  # Adjust as needed
mask[y_freq_to_remove-band_width:y_freq_to_remove+band_width, :] = 0
mask[rows-y_freq_to_remove-band_width:rows-y_freq_to_remove+band_width, :] = 0

# Apply the mask to the shifted Fourier Transform
f_transform_shifted *= mask

# Inverse Fourier Transform to get the modified image
f_ishift = np.fft.ifftshift(f_transform_shifted)
filtered_image = np.abs(np.fft.ifft2(f_ishift))

# Display the original and filtered images
plt.figure(figsize=(12, 6), dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Removed Specific Frequency)')

cv2.imwrite('filtered_image.png', filtered_image)

power_spec = power_spectrum(filtered_image)
# plt.figure(dpi=300)
# plt.imshow(np.log(1 + power_spec))  # Visualize power spectrum
# plt.show()

# Normalize the power spectrum
# Use percentile clipping to limit the range of values
pmin, pmax = np.percentile(power_spec, [1, 99])
power_spec_clipped = np.clip(power_spec, pmin, pmax)

plt.figure(dpi=300)
plt.imshow(np.log1p(power_spec_clipped), cmap='viridis')  # Visualize power spectrum
plt.colorbar()
plt.title('Power Spectrum')
plt.show()
    