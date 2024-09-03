import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pypylon import pylon
import numpy as np

# List of classification annotations
annotations = [
    "6024_PASS", "6038_PASS", "7024_PASS", "7038_PASS",
    "6024_FAIL", "6038_FAIL", "7024_FAIL", "7038_FAIL"
]

# Directory to save images
output_dir = "annotated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize image counter
image_counter = 0

def initialize_camera():
    try:
        # Create an instant camera object with the camera device found by Pylon
        tlf = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tlf.CreateFirstDevice())
        camera.Open()
        return camera
    except Exception as e:
        print(f"Error: {e}")
        return None
    
camera = initialize_camera()

# Open the camera once and keep it open
#cap = cv2.VideoCapture(0)

# Initialize image counter based on existing images
def initialize_image_counter():
    max_number = 0
    for annotation in annotations:
        annotation_dir = os.path.join(output_dir, annotation)
        if not os.path.exists(annotation_dir):
            os.makedirs(annotation_dir)
            continue
        
        for filename in os.listdir(annotation_dir):
            if filename.startswith("image_") and filename.endswith(".jpg"):
                number = int(filename[6:11])  # Extract the number part
                max_number = max(max_number, number)
    
    return max_number

image_counter = initialize_image_counter()

def get_next_image_filename(annotation):
    global image_counter
    image_filename = None
    while True:
        image_counter += 1
        filename = f"image_{image_counter:05d}.jpg"
        image_filename = os.path.join(output_dir, annotation, filename)
        if not os.path.exists(image_filename):
            break
    return image_filename

def capture_image():
    if not camera:
        print("Error: Camera not initialized.")
        return None
    
    try:
        # Start grabbing images
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grab_result.GrabSucceeded():
            # Convert image to numpy array
            img = grab_result.Array
            
            # # Check the format of the image
            # img_format = grab_result.GetPixelType()
            
            # # Example for handling different formats:
            # if img_format == pylon.PixelType_Mono8:
            #     print("Detected grayscale format.")
            #     # Image is already in grayscale, no conversion needed
            # elif img_format == pylon.PixelType_RGB8packed:
            #     print("Detected RGB format.")
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            # elif img_format == pylon.PixelType_BayerRG8:
            #     print("Detected Bayer format.")
            #     img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)  # Convert Bayer to BGR
            
            img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
            grab_result.Release()
            return img
        else:
            print("Error: Could not grab image.")
            grab_result.Release()
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Ensure that grabbing is stopped after image capture
        camera.StopGrabbing()

#def capture_image():
#    if not cap.isOpened():
#        print("Error: Could not open camera.")
#        return None
#    
#    ret, frame = cap.read()
#
#    if not ret:
#        print("Error: Could not capture image.")
#        return None
#    
#    return frame

def save_image_with_annotation(image, annotation):
    annotation_dir = os.path.join(output_dir, annotation)
    os.makedirs(annotation_dir, exist_ok=True)
    
    image_filename = get_next_image_filename(annotation)
    
    # Save the image
    cv2.imwrite(image_filename, image)
    print(f"Image saved as {image_filename}")

    return image_filename

def update_display(image_filename):
    img = Image.open(image_filename)
    img = img.resize((500, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    
    display_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
    display_label.config(image=img_tk)

def on_take_picture():
    # Capture image
    image = capture_image()
    if image is None:
        return
    
    # Get selected annotation
    annotation = annotation_var.get()
    
    # Save image with annotation
    image_filename = save_image_with_annotation(image, annotation)
    
    # Update display
    update_display(image_filename)

# Tkinter GUI setup
root = tk.Tk()
root.title("QC Image Annotation")

root.minsize(600,450)

# Dropdown menu for annotations
annotation_var = tk.StringVar(value=annotations[0])
annotation_menu = ttk.Combobox(root, textvariable=annotation_var, values=annotations, state="readonly")
annotation_menu.pack(pady=10)

# Take Picture button
take_picture_button = tk.Button(root, text="Take Picture", command=on_take_picture)
take_picture_button.pack(pady=10)

# Display area for the captured image
display_label = tk.Label(root)
display_label.pack(pady=10)

# Run the GUI loop
root.mainloop()

# Release the camera when the application is closed
#cap.release()
if camera:
    camera.StopGrabbing()
    camera.Close()