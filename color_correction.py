from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import cv2
import sys

def find_color_card(image):
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    return corners, ids

def overlay_markers(image, corners, ids):
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image

def extract_color_card(image, corners, ids):    
    try:
        ids = ids.flatten()
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]
    except:
        return None
    
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    return card

def apply_lab_color_correction(input_image, ref_card, input_card):
    # Convert the cards and image to LAB color space
    ref_card_lab = cv2.cvtColor(ref_card, cv2.COLOR_BGR2LAB)
    input_card_lab = cv2.cvtColor(input_card, cv2.COLOR_BGR2LAB)
    input_image_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

    # Calculate the mean and standard deviation of each channel in LAB space
    ref_mean, ref_std = cv2.meanStdDev(ref_card_lab)
    input_mean, input_std = cv2.meanStdDev(input_card_lab)

    # Apply the correction
    l, a, b = cv2.split(input_image_lab)
    l = ((l - input_mean[0]) / input_std[0]) * ref_std[0] + ref_mean[0]
    a = ((a - input_mean[1]) / input_std[1]) * ref_std[1] + ref_mean[1]
    b = ((b - input_mean[2]) / input_std[2]) * ref_std[2] + ref_mean[2]

    # Clip the values to ensure they stay within the valid LAB range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge channels back
    corrected_lab = cv2.merge((l, a, b))
    corrected_image = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return corrected_image


# Path to the images
refimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/color_ref.png'
inputimage = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/color_target.png'

# Load and resize the images
print("[INFO] loading images...")
ref = cv2.imread(refimage)
image = cv2.imread(inputimage)
ref = cv2.resize(ref, (600, int(ref.shape[0] * (600.0 / ref.shape[1]))))
image = cv2.resize(image, (900, int(image.shape[0] * (900.0 / image.shape[1]))))

# Find and extract color matching cards
print("[INFO] finding color matching cards...")
refCorners, refIds = find_color_card(ref)
imageCorners, imageIds = find_color_card(image)

# Overlay markers
ref_with_markers = overlay_markers(ref.copy(), refCorners, refIds)
image_with_markers = overlay_markers(image.copy(), imageCorners, imageIds)
cv2.imshow("Reference with Markers", ref_with_markers)
cv2.imshow("Input with Markers", image_with_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract color matching cards
refCard = extract_color_card(ref, refCorners, refIds)
imageCard = extract_color_card(image, imageCorners, imageIds)

# Exit if color cards are not found
if refCard is None or imageCard is None:
    print("[INFO] could not find color matching card in both images")
    sys.exit(0)

cv2.imshow("Reference Color Card", cv2.resize(refCard, (400, int(refCard.shape[0] * (400.0 / refCard.shape[1])))))
cv2.imshow("Input Color Card", cv2.resize(imageCard, (400, int(imageCard.shape[0] * (400.0 / imageCard.shape[1])))))

# Apply LAB color correction to the entire image
print("[INFO] applying LAB color correction...")
corrected_image = apply_lab_color_correction(image, refCard, imageCard)

# Show the corrected image
cv2.imshow("Corrected Image", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
