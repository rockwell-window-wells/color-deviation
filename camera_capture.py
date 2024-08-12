# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:47:48 2024

@author: Ryan.Larson
"""

import cv2
import time

start = time.time()
cam_port = 0
cam = cv2.VideoCapture(cam_port)
end = time.time()
print(f"Initialization time:\t{end-start}")

start = time.time()
result, image = cam.read()
end = time.time()
print(f"Capture time:\t{end-start}")

if result:
    # cv2.imshow("Image", image)
    cv2.imwrite("Webcam_image.png", image)
    
    # cv2.waitKey(0)
    # cv2.destroyWindow("Image")

else:
    print("No image detected. Try again.")