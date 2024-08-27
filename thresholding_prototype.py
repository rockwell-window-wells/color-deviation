import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/test_image (1).jpg'
# image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (21, 21), 0)

# Apply adaptive thresholding to segment the image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(10, 5),dpi=300)
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Thresholded Image")
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()






















# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 29 15:50:09 2024

# @author: Ryan.Larson
# """

# import cv2
# import numpy as np

# # Load the image
# imfile = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png'
# # image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

# # # Apply thresholding
# # thresh = 127    # The threshold value. This is the value against which each pixel value is compared.
# # maxval = 255    # The maximum value assigned to the pixels that satisfy the condition set by the threshold type. For binary and binary inverted thresholding, this value is used for pixels meeting the criteria.
# # _, segmented_image = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)


# # Load the image
# image = cv2.imread(imfile)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# # Apply thresholding
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Remove small noise and fill small holes
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# # Background area
# sure_bg = cv2.dilate(opening, kernel, iterations=3)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)

# # Marker labeling
# _, markers = cv2.connectedComponents(sure_fg)

# # Add one to all labels so that sure background is not 0, but 1
# markers = markers + 1

# # Now, mark the region of unknown with zero
# markers[unknown == 0] = 0

# markers = cv2.watershed(image, markers)
# image[markers == -1] = [255, 0, 0]

# # # Display the result
# # cv2.imshow('Watershed Segmented Image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Get screen resolution (example resolution: 1920x1080)
# screen_width = 1920
# screen_height = 1080
# # Define a margin ratio (e.g., 0.9 means 10% margin)
# margin_ratio = 0.9
# # Adjust the available space by the margin ratio
# adjusted_screen_width = screen_width * margin_ratio
# adjusted_screen_height = screen_height * margin_ratio
# # Determine the scale factor
# scale_factor = min(adjusted_screen_width / image.shape[1], adjusted_screen_height / image.shape[0])
# # Resize the image to fit the screen with margin
# resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# # # Display the result
# # cv2.imshow('Watershed Segmented Image', resized_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# markers[unknown == 0] = 0
# markers = cv2.watershed(image, markers)
# image[markers == -1] = [255, 0, 0]  # Mark boundaries with red

# # Visualize markers
# markers_8u = (markers.astype('uint8') * 10) % 255  # To visualize different regions with different intensities
# resized_image = cv2.resize(markers_8u, None, fx=scale_factor, fy=scale_factor)
# cv2.imshow('Markers', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()