# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:43:22 2024

@author: Ryan.Larson
"""

import base64
import json

import labelme
import matplotlib.pyplot as plt
import numpy as np


# load annoated file
json_file = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/test_training/annotations/20240806_135622.json'
with open(json_file, "r") as f:
    data = json.load(f)

# load image from data["imageData"]
image = labelme.utils.img_b64_to_arr(data["imageData"])
print("image:", image.shape, image.dtype)

# load label_names, label, label_points_xy from data["shapes"]
unique_label_names = ["_background_"] + sorted(
    set([shape["label"] for shape in data["shapes"]])
)
label = np.zeros(image.shape[:2], dtype=np.int32)
label_names = []
label_points_xy = []
for shape in data["shapes"]:
    label_id = unique_label_names.index(shape["label"])

    label_names.append(shape["label"])

    mask = labelme.utils.shape_to_mask(
        img_shape=image.shape[:2],
        points=shape["points"],
        shape_type=shape["shape_type"],
    )
    label[mask] = label_id
    label_points_xy.append(shape["points"])
print("label:", label.shape, label.dtype)
print("label_names:", label_names)

# visualize
colors = "rgbymc"
#
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.imshow(label, alpha=0.5)
#
plt.subplot(1, 2, 2)
plt.imshow(image)
for i, (label_name, label_points_xy_i) in enumerate(zip(label_names, label_points_xy)):
    label_id = unique_label_names.index(label_name)
    label_points_xy_i = np.array(label_points_xy_i)
    plt.plot(
        label_points_xy_i[:, 0],
        label_points_xy_i[:, 1],
        marker="o",
        color=colors[label_id % len(colors)],
        label=label_name if label_name not in label_names[:i] else None,
    )
plt.legend()
#
plt.tight_layout()
plt.suptitle(json_file)
plt.show()