# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:06:51 2024

@author: Ryan.Larson
"""

from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import sys

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect the
    # markers in the input image
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    
    # return the detected corners and IDs for overlay
    return corners, ids

def overlay_markers(image, corners, ids):
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image

def extract_color_card(image, corners, ids):
    try:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        
        # extract the top-left marker
        i = np.squeeze(np.where(ids == 100))
        topLeft = np.squeeze(corners[i])[0]
        
        # extract the top-right marker
        i = np.squeeze(np.where(ids == 200))
        topRight = np.squeeze(corners[i])[1]
        
        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 300))
        bottomRight = np.squeeze(corners[i])[2]
        
        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 400))
        bottomLeft = np.squeeze(corners[i])[3]
        
    except:
        return None
    
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    
    return card

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-r", "--reference", required=True, help="path to the input reference image")
# ap.add_argument("-i", "--input", required=True, help="path to the input image to apply color correction to")
# args = vars(ap.parse_args())

refimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/color_correction_ref.jpg'
inputimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/color_correction_input.jpg'

# load the reference image and input images from disk
print("[INFO] loading images...")
ref = cv2.imread(refimage)
image = cv2.imread(inputimage)
# ref = cv2.imread(args["reference"])
# image = cv2.imread(args["input"])

# resize the reference and input images
ref = imutils.resize(ref, width=600)
image = imutils.resize(image, width=600)

# find the color matching card in each image
print("[INFO] finding color matching cards...")
refCorners, refIds = find_color_card(ref)
imageCorners, imageIds = find_color_card(image)

# overlay markers on the images
ref_with_markers = overlay_markers(ref.copy(), refCorners, refIds)
image_with_markers = overlay_markers(image.copy(), imageCorners, imageIds)

# display the images with detected markers
cv2.imshow("Reference with Markers", ref_with_markers)
cv2.imshow("Input with Markers", image_with_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# extract the color matching card in each image
refCard = extract_color_card(ref, refCorners, refIds)
imageCard = extract_color_card(image, imageCorners, imageIds)

# if the color matching card is not found in either the reference image or the
# input image, gracefully exit
if refCard is None or imageCard is None:
    print("[INFO] could not find color matching card in both images")
    sys.exit(0)

# show the color matching card in the reference image and input image, respectively
cv2.imshow("Reference Color Card", refCard)
cv2.imshow("Input Color Card", imageCard)

# apply histogram matching from the color matching card in the reference image
# to the color matching card in the input image
print("[INFO] matching images...")
imageCard = exposure.match_histograms(imageCard, refCard, multichannel=True)

# show our input color matching card after histogram matching
cv2.imshow("Input Color Card After Matching", imageCard)
cv2.waitKey(0)
cv2.destroyAllWindows()




# # -*- coding: utf-8 -*-
# """
# Created on Mon Aug  5 14:06:51 2024

# @author: Ryan.Larson
# """

# from imutils.perspective import four_point_transform
# from skimage import exposure
# import numpy as np
# import argparse
# import imutils
# import cv2
# import sys

# def find_color_card(image):
#     # load the ArUCo dictionary, grab the ArUCo parameters, and detect the
#     # markers in the input image
#     arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#     arucoParams = cv2.aruco.DetectorParameters()
#     (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    
#     # try to extract the coordinates of the color correction card
#     try:
#         # Draw detected markers and their IDs on the image
#         if ids is not None:
#             cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
#         # otherwise, we've found the four ArUco markers, so we can continue
#         # by flattening the ArUco IDs list
#         ids = ids.flatten()
        
#         # extract the top-left marker
#         i = np.squeeze(np.where(ids == 100))
#         topLeft = np.squeeze(corners[i])[0]
        
#         # extract the top-right marker
#         i = np.squeeze(np.where(ids == 200))
#         topRight = np.squeeze(corners[i])[1]
        
#         # extract the bottom-right marker
#         i = np.squeeze(np.where(ids == 300))
#         bottomRight = np.squeeze(corners[i])[2]
        
#         # extract the bottom-left marker
#         i = np.squeeze(np.where(ids == 400))
#         bottomLeft = np.squeeze(corners[i])[3]
        
#     # if no color correction card is found, return None
#     except:
#         return None
    
#     # build the list of reference points and apply a perspective transform
#     # to obtain a top-down, bird's-eye view of the color matching card
#     cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
#     card = four_point_transform(image, cardCoords)
    
#     # return the color matching card to the calling function
#     return card

# # # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-r", "--reference", required=True, help="path to the input reference image")
# # ap.add_argument("-i", "--input", required=True, help="path to the input image to apply color correction to")
# # args = vars(ap.parse_args())

# refimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/color_correction_ref.jpg'
# inputimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/color_correction_input.jpg'

# # load the reference image and input images from disk
# print("[INFO] loading images...")
# ref = cv2.imread(refimage)
# image = cv2.imread(inputimage)
# # ref = cv2.imread(args["reference"])
# # image = cv2.imread(args["input"])

# # resize the reference and input images
# ref = imutils.resize(ref, width=600)
# image = imutils.resize(image, width=600)

# # display the reference and input images to the screen
# cv2.imshow("Reference", ref)
# cv2.imshow("Input", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # find the color matching card in each image
# print("[INFO] finding color matching cards...")
# refCard = find_color_card(ref)
# imageCard = find_color_card(image)

# # if the color matching card is not found in either the reference image or the
# # input image, gracefully exit
# if refCard is None or imageCard is None:
#     print("[INFO] could not find color matching card in both images")
#     sys.exit(0)
    
# # show the color matching card in the reference image and input image, respectively
# cv2.imshow("Reference Color Card", refCard)
# cv2.imshow("Input Color Card", imageCard)

# # apply histogram matching from the color matching card in the reference image
# # to the color matching card in the input image
# print("[INFO] matching images...")
# imageCard = exposure.match_histograms(imageCard, refCard, multichannel=True)

# # show our input color matching card after histogram matching
# cv2.imshow("Input Color Card After Matching", imageCard)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
