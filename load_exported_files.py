# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:47:21 2024

@author: Ryan.Larson
"""

import numpy as np
import PIL.Image

import matplotlib.pyplot as plt


# load exported files
directory = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/test_training/annotations/20240806_135622/'
image = np.asarray(PIL.Image.open(directory+"img.png"))
label = np.asarray(PIL.Image.open(directory+"label.png"))
with open(directory+"label_names.txt", "r") as f:
    label_names = f.read().splitlines()

# extract masks from label
masks = {}
for label_id, label_name in enumerate(label_names):
    mask = label == label_id
    masks[(label_id, label_name)] = mask

# print stats
print("image:", image.shape, image.dtype)
print("label:", label.shape, label.dtype)
print("label_names:", label_names)

# visualize
rows = 2
columns = max(2, len(label_names))
#
plt.subplot(rows, columns, 1)
plt.title("image")
plt.imshow(image)
#
plt.subplot(rows, columns, 2)
plt.title("label")
plt.imshow(label)
#
plt.subplot(rows, columns, 3)
plt.title("label overlaid")
plt.imshow(image)
plt.imshow(label, alpha=0.5)
#
for (label_id, label_name), mask in masks.items():
    plt.subplot(rows, columns, 4 + label_id)
    plt.title(f"{label_id}:{label_name}")
    plt.imshow(mask, cmap="gray")
#
plt.tight_layout()
plt.show()