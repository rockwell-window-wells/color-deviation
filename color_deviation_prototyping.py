# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:43:11 2024

@author: Ryan.Larson
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def calculate_histogram(image):
    """Calculate the color histogram of an image in LAB color space."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_hist = cv2.calcHist([lab_image[..., 0]], [0], None, [256], [0, 256])
    a_hist = cv2.calcHist([lab_image[..., 1]], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([lab_image[..., 2]], [0], None, [256], [0, 256])
    return l_hist.flatten(), a_hist.flatten(), b_hist.flatten()

def calculate_deviation(lab_image, reference_color):
    """Calculate the minimum and maximum deviation from the reference LAB color."""
    l_ref, a_ref, b_ref = reference_color
    # # Normalize the channels
    # lab_array = np.asarray(lab_image)
    # lab_array = lab_array.astype('float32')
    
    # L_channel = lab_array[:, :, 0] * (100.0 / 255.0)
    # a_channel = lab_array[:, :, 1] - 128.0
    # b_channel = lab_array[:, :, 2] - 128.0
    
    # # Combine channels into a Lab image if needed
    # lab_image_update = np.zeros_like(lab_image, dtype=np.float32)
    # lab_image_update[:,:,0] = L_channel.astype(np.float32)
    # lab_image_update[:,:,1] = a_channel.astype(np.float32)
    # lab_image_update[:,:,2] = b_channel.astype(np.float32)
    # image_lab_normalized = cv2.merge([L_channel, a_channel, b_channel])
    l_deviation = lab_image[..., 0] - l_ref
    a_deviation = lab_image[..., 1] - a_ref
    b_deviation = lab_image[..., 2] - b_ref
    delta_e = np.sqrt(l_deviation**2 + a_deviation**2 + b_deviation**2)
    # min_deviation = np.min(delta_e)
    # max_deviation = np.max(delta_e)
    deviations = (l_deviation, a_deviation, b_deviation)
    return delta_e, deviations

def find_max_deviation_points(delta_e):
    """Find the points of maximum deviation in the LAB image."""
    max_idx = np.unravel_index(np.argmax(delta_e), delta_e.shape)
    return max_idx

def create_overlay(delta_e, max_deviation):
    """Create an overlay image with colors mapped from green to red based on delta E values."""
    normalized_delta_e = delta_e / max_deviation
    overlay = np.zeros((*delta_e.shape, 3), dtype=np.uint8)

    # Green to red gradient: (0, 255, 0) to (255, 0, 0)
    B, G, R = green_yellow_red(normalized_delta_e)
    overlay[..., 0] = B     # Blue channel
    overlay[..., 1] = G  # Green channel (decreases with delta E)
    overlay[..., 2] = R       # Red channel (increases with delta E)
    # overlay[..., 0] = 0     # Blue channel
    # overlay[..., 1] = 255 * (1 - normalized_delta_e)  # Green channel (decreases with delta E)
    # overlay[..., 2] = 255 * normalized_delta_e       # Red channel (increases with delta E)

    return overlay

def green_yellow_red(normalized_val):
    """Map values from 0 to 1 to a continuous gradient from green to yellow to red."""
    # Ensure normalized_val is a NumPy array
    normalized_val = np.asarray(normalized_val)
    
    # Initialize B, G, R arrays
    B = np.zeros_like(normalized_val, dtype=np.uint8)
    G = np.zeros_like(normalized_val, dtype=np.uint8)
    R = np.zeros_like(normalized_val, dtype=np.uint8)
    
    # Green to yellow gradient (0 to 0.5)
    mask1 = normalized_val <= 0.5
    R[mask1] = (normalized_val[mask1] * 2 * 255).astype(np.uint8)
    G[mask1] = 255

    # Yellow to red gradient (0.5 to 1.0)
    mask2 = normalized_val > 0.5
    R[mask2] = 255
    G[mask2] = (255 - (normalized_val[mask2] - 0.5) * 2 * 255).astype(np.uint8)

    return B, G, R

def calculate_max_neighbor_deviation(lab_image, neighbor_size=1):
    """
    Calculate the maximum delta E for each pixel based on its neighbors.
    
    Parameters:
        lab_image (np.ndarray): The input Lab image.
        neighbor_size (int): The half-width of the neighborhood to consider. For a 4x4 neighborhood, use 2.
        
    Returns:
        np.ndarray: Array of maximum delta E values for each pixel based on its neighbors.
    """
    h, w, _ = lab_image.shape
    max_delta_e = np.zeros((h, w), dtype=np.float32)

    for i in range(neighbor_size, h - neighbor_size):
        for j in range(neighbor_size, w - neighbor_size):
            center = lab_image[i, j]
            neighbors = lab_image[i-neighbor_size:i+neighbor_size+1, j-neighbor_size:j+neighbor_size+1].reshape(-1, 3)

            # Exclude the center pixel from the neighbor list
            mask = ~np.all(neighbors == center, axis=1)
            neighbors = neighbors[mask]

            if neighbors.size > 0:
                delta_e = np.sqrt(np.sum((neighbors - center) ** 2, axis=1))
                max_delta_e[i, j] = np.max(delta_e)
            else:
                max_delta_e[i, j] = 0  # or np.nan if you prefer to mark these pixels differently

    return max_delta_e

def plot_LAB_histogram(lab_image):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)
    
    L_values = lab_image[:,:,0].flatten() * 100.0 / 255.0
    a_values = lab_image[:,:,1].flatten()
    b_values = lab_image[:,:,2].flatten()
    
    L_hist, L_bin_edges = np.histogram(L_values, bins=100)
    a_hist, a_bin_edges = np.histogram(a_values, bins=100)
    b_hist, b_bin_edges = np.histogram(b_values, bins=100)
    
    plt.plot(L_bin_edges[:-1], L_hist, drawstyle='steps-post', color='blue', linewidth=2)
    plt.plot(a_bin_edges[:-1], a_hist, drawstyle='steps-post', color='green', linewidth=2)
    plt.plot(b_bin_edges[:-1], b_hist, drawstyle='steps-post', color='red', linewidth=2)
    
    plt.fill_between(L_bin_edges[:-1], L_hist, step='post', alpha=0.4, color='blue')
    plt.fill_between(a_bin_edges[:-1], a_hist, step='post', alpha=0.4, color='green')
    plt.fill_between(b_bin_edges[:-1], b_hist, step='post', alpha=0.4, color='red')
    
    
    # sns.histplot(L_values, kde=True, color="blue", bins=10, alpha=0.4, edgecolor="blue", linewidth=2, label="L")
    # sns.histplot(a_values, kde=True, color="green", bins=10, alpha=0.4, edgecolor="green", linewidth=2, label="a")
    # sns.histplot(b_values, kde=True, color="red", bins=10, alpha=0.4, edgecolor="red", linewidth=2, label="b")
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Lab Histogram')
    # plt.title('Lab Histogram with KDE')
    
    plt.show()
    
    
def plot_delta_E_histogram(delta_e):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=300)
    
    delta_e_flat = delta_e.flatten()
    hist, bin_edges = np.histogram(delta_e_flat, bins=30)
    plt.plot(bin_edges[:-1], hist, drawstyle='steps-post', color='blue', linewidth=2)
    plt.fill_between(bin_edges[:-1], hist, step='post', alpha=0.4, color='blue')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Delta E Histogram')
    plt.show()
    

# def green_yellow_red(normalized_val):
#     """Values from 0 to 0.5 are returned as green to yellow, and values from
#     0.5 to 1.0 are returned as yellow to red."""
#     # Ensure normalized_val is a NumPy array
#     normalized_val = np.asarray(normalized_val)

#     # Initialize B, G, R arrays with zeros
#     B = np.zeros_like(normalized_val, dtype=np.uint8)
#     G = np.zeros_like(normalized_val, dtype=np.uint8)
#     R = np.zeros_like(normalized_val, dtype=np.uint8)

#     # Define masks for different ranges
#     mask1 = normalized_val <= 0.5
#     mask2 = normalized_val > 0.5

#     # Apply the color transformation for the first range (green to yellow)
#     R[mask1] = 255 * 2 * normalized_val[mask1]
#     G[mask1] = 255

#     # Apply the color transformation for the second range (yellow to red)
#     R[mask2] = 255
#     G[mask2] = 255 * (1 - 2 * (normalized_val[mask2] - 0.5))

#     return B, G, R

# def main(image_path, reference_lab):
#     """Main function to process the image and compute deviations."""
#     image = cv2.imread(image_path)
#     image_float32 = np.float32(image) / 255.0
#     # l_hist, a_hist, b_hist = calculate_histogram(image)
#     # fig = plt.figure(dpi=300)
#     # plt.plot(l_hist, label="L")
#     # plt.plot(a_hist, label="a")
#     # plt.plot(b_hist, label="b")
#     # plt.legend()
#     # plt.show()

#     lab_image = cv2.cvtColor(image_float32, cv2.COLOR_BGR2Lab)
#     max_delta_e = calculate_max_neighbor_deviation(lab_image)
#     min_deviation, max_deviation, delta_e = calculate_deviation(lab_image, reference_lab)
#     print("Minimum deviation from reference color:", min_deviation)
#     print("Maximum deviation from reference color:", max_deviation)

#     max_idx = find_max_deviation_points(delta_e)
#     print("Maximum deviation at:", max_idx)
    
#     # Filter delta E values above 10
#     delta_e[delta_e > 10] = np.nan

#     cmap = plt.get_cmap('jet')
#     fig, ax = plt.subplots(dpi=300)
#     cax = ax.imshow(delta_e, cmap=cmap, interpolation='bilinear')
#     cbar = plt.colorbar(cax, ax=ax)
#     cbar.set_label('Delta E')

#     # overlay = create_overlay(delta_e, max_deviation)
#     # # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

#     # # Blend the original image with the overlay
#     # alpha = 0.4  # Transparency factor
#     # blended_image = cv2.addWeighted(image, 1, overlay, alpha, 0)

#     # # Get screen resolution (example resolution: 1920x1080)
#     # screen_width = 1920
#     # screen_height = 1080
#     # # Define a margin ratio (e.g., 0.9 means 10% margin)
#     # margin_ratio = 0.9
#     # # Adjust the available space by the margin ratio
#     # adjusted_screen_width = screen_width * margin_ratio
#     # adjusted_screen_height = screen_height * margin_ratio
#     # # Determine the scale factor
#     # scale_factor = min(adjusted_screen_width / blended_image.shape[1], adjusted_screen_height / blended_image.shape[0])
#     # # Resize the image to fit the screen with margin
#     # resized_image = cv2.resize(blended_image, None, fx=scale_factor, fy=scale_factor)

#     # cv2.imshow("Annotated Image", resized_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     # Optionally save the annotated image
#     # cv2.imwrite("annotated_image.png", resized_image)

if __name__ == "__main__":
    # Example usage
    image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/lighting_test (4).jpg'  # Update with your image path
    # image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI_masked.png'  # Update with your image path
    # image_path = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/reference_images/ROI.png'  # Update with your image path
    # reference_lab = (90.0, 5.33, 1.47)  # Dummy reference color
    # reference_lab = (76.26, -1.54, -1.12)  # left side reference color (white-ish area)
    # reference_lab = (77.65, -0.36, 0.997)  # left side reference color (appears as main color)
    reference_lab = (65.431, 2.298, 77.846)  # reference color for test_image (1).jpg
    # reference_lab = (79.43, 5.33, 1.47)  # A&P reference color

    # """Main function to process the image and compute deviations."""
    # bgra_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # bgr_image = bgra_image[:, :, :3]
    # bgr_image = np.float32(bgr_image) / 255.0
    # alpha_channel = bgra_image[:, :, 3]
    # alpha_channel = np.float32(alpha_channel) / 255.0
    # alpha_channel[alpha_channel < 1.0] = 0.0
    
    bgr_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    bgr_image = np.float32(bgr_image) / 255.0
    
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    
    # mask = alpha_channel > 0
    
    # lab_image_masked = lab_image.copy()
    # lab_image_masked[~mask] = [0, 0, 0]
    
    neighbor_size = 2
    # max_delta_e = calculate_max_neighbor_deviation(lab_image, neighbor_size)
    delta_e, deviations = calculate_deviation(lab_image, reference_lab)
    # max_delta_e = calculate_max_neighbor_deviation(lab_image_masked, neighbor_size)
    # min_deviation, max_deviation, delta_e = calculate_deviation(lab_image_masked, reference_lab)
    
    # 3D scatter plot of deviations
    L_deviation = deviations[0]
    a_deviation = deviations[1]
    b_deviation = deviations[2]
    L_deviation = L_deviation.flatten()
    a_deviation = a_deviation.flatten()
    b_deviation = b_deviation.flatten()
    
    data = {"L deviation": L_deviation,
            "a deviation": a_deviation,
            "b deviation": b_deviation}
    df = pd.DataFrame(data)
    df['Delta E'] = np.sqrt(df['L deviation']**2 + df['a deviation']**2 + df['b deviation']**2)
    fig = px.scatter_3d(df.sample(n=5000), x="a deviation", y="b deviation", z = "L deviation", color='Delta E')
    fig.show(renderer='browser')
    
    # Filter delta E values above threshold
    # thresh = 60
    # delta_e[delta_e > thresh] = np.nan
    # max_delta_e[max_delta_e > thresh] = np.nan
    
    # cmap = plt.get_cmap('jet')
    # fig, ax = plt.subplots(dpi=300)
    # cax = ax.imshow(delta_e, cmap=cmap, interpolation='bilinear')
    # cbar = plt.colorbar(cax, ax=ax)
    # cbar.set_label('Delta E')
    
    # fig, ax = plt.subplots(dpi=300)
    # cax = ax.imshow(max_delta_e, cmap=cmap, interpolation='bilinear')
    # cbar = plt.colorbar(cax, ax=ax)
    # cbar.set_label('Max Neighboring Delta E')
    
    # plot_LAB_histogram(lab_image)
    # plot_delta_E_histogram(delta_e)


    
    # # image = cv2.imread(image_path)
    # # image_float32 = np.float32(image) / 255.0
    # # l_hist, a_hist, b_hist = calculate_histogram(image)
    # # fig = plt.figure(dpi=300)
    # # plt.plot(l_hist, label="L")
    # # plt.plot(a_hist, label="a")
    # # plt.plot(b_hist, label="b")
    # # plt.legend()
    # # plt.show()

    # lab_image = cv2.cvtColor(image_float32, cv2.COLOR_BGR2Lab)
    # neighbor_size=6
    # # max_delta_e = calculate_max_neighbor_deviation(lab_image, neighbor_size)
    # min_deviation, max_deviation, delta_e = calculate_deviation(lab_image, reference_lab)
    # # print("Minimum deviation from reference color:", min_deviation)
    # # print("Maximum deviation from reference color:", max_deviation)

    # # max_idx = find_max_deviation_points(delta_e)
    # # print("Maximum deviation at:", max_idx)
    
    # # Filter delta E values above 10
    # # delta_e[delta_e > 10] = np.nan
    # # max_delta_e[max_delta_e > 20] = np.nan

    # cmap = plt.get_cmap('jet')
    # fig, ax = plt.subplots(dpi=300)
    # cax = ax.imshow(delta_e, cmap=cmap, interpolation='bilinear')
    # cbar = plt.colorbar(cax, ax=ax)
    # cbar.set_label('Delta E')
    
    # # fig, ax = plt.subplots(dpi=300)
    # # cax = ax.imshow(max_delta_e, cmap=cmap, interpolation='bilinear')
    # # cbar = plt.colorbar(cax, ax=ax)
    # # cbar.set_label('Max Neighboring Delta E')







    # overlay = create_overlay(delta_e, max_deviation)
    # # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # # Blend the original image with the overlay
    # alpha = 0.4  # Transparency factor
    # blended_image = cv2.addWeighted(image, 1, overlay, alpha, 0)

    # # Get screen resolution (example resolution: 1920x1080)
    # screen_width = 1920
    # screen_height = 1080
    # # Define a margin ratio (e.g., 0.9 means 10% margin)
    # margin_ratio = 0.9
    # # Adjust the available space by the margin ratio
    # adjusted_screen_width = screen_width * margin_ratio
    # adjusted_screen_height = screen_height * margin_ratio
    # # Determine the scale factor
    # scale_factor = min(adjusted_screen_width / blended_image.shape[1], adjusted_screen_height / blended_image.shape[0])
    # # Resize the image to fit the screen with margin
    # resized_image = cv2.resize(blended_image, None, fx=scale_factor, fy=scale_factor)

    # cv2.imshow("Annotated Image", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Optionally save the annotated image
    # cv2.imwrite("annotated_image.png", resized_image)
