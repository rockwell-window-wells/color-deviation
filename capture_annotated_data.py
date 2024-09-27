#!/home/rwengineering/Documents/GitHub/color-deviation/venv/bin/python3
import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pypylon import pylon
import numpy as np
from process_annotated_data import process_main_directory

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
last_image_filename = None  # Store the last image filename globally

def initialize_camera():
    try:
        # Create an instant camera object with the camera device found by Pylon
        tlf = pylon.TlFactory.GetInstance()
        camera = pylon.InstantCamera(tlf.CreateFirstDevice())
        camera.Open()
        
        print_available_values(camera, "GainAuto")
        print_available_values(camera, "ExposureAuto")
        
        configure_camera(camera)
        
        return camera
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def print_available_values(camera, feature_name):
    try:
        node = getattr(camera, feature_name)
        available_values = node.Symbolics
        print(f"Available values for {feature_name}: {available_values}")
    except Exception as e:
        print(f"Error querying {feature_name}: {e}")

def configure_camera(camera):
    try:
        # camera.GainAuto.Value = "On"
        # camera.ExposureAuto.Value = "On"
        # camera.ExposureTime.Value = 20000   # Exposure time in microseconds
        
        # camera.BalanceWhiteAuto.Value = "Off"
        # camera.BalanceWhiteRed.Value = 1.0
        # camera.BalanceWhiteGreen.Value = 1.0
        # camera.BalanceWhiteBlue.Value = 1.0
        
        camera.GainAuto.SetValue("Off")
        camera.Gain.SetValue(0.0)
        camera.ExposureAuto.SetValue("Continuous")
        camera.ExposureTime.SetValue(10000)     # Exposure time in microseconds
        
        camera.BalanceWhiteAuto.SetValue("Off")
        camera.BalanceWhiteRed.SetValue(1.0)
        camera.BalanceWhiteGreen.SetValue(1.0)
        camera.BalanceWhiteBlue.SetValue(1.0)
        
    except Exception as e:
        print(f"Error configuring exposure: {e}")

camera = initialize_camera()

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
        camera.StopGrabbing()

def save_image_with_annotation(image, annotation):
    global last_image_filename
    annotation_dir = os.path.join(output_dir, annotation)
    os.makedirs(annotation_dir, exist_ok=True)
    
    image_filename = get_next_image_filename(annotation)
    
    # Save the image
    cv2.imwrite(image_filename, image)
    print(f"Image saved as {image_filename}")

    last_image_filename = image_filename  # Store the last image filename
    return image_filename

def update_display(image_filename):
    img = Image.open(image_filename)
    img = img.resize((500, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    
    display_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
    display_label.config(image=img_tk)

def clear_display():
    display_label.config(image='')

def show_success_message(image_filename):
    image_name = os.path.basename(image_filename)
    success_message.config(text=f"{image_name} has been taken")
    root.after(10000, clear_success_message)  # Clear the message after 10 seconds

def show_delete_message(image_filename):
    image_name = os.path.basename(image_filename)
    success_message.config(text=f"{image_name} has been deleted")
    root.after(5000, clear_success_message)  # Clear the message after 5 seconds

def clear_success_message():
    success_message.config(text="")

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
    
    # Show success message
    show_success_message(image_filename)

def on_reject_picture():
    global last_image_filename
    
    if last_image_filename:
        try:
            os.remove(last_image_filename)
            print(f"Deleted {last_image_filename}")
            show_delete_message(last_image_filename)
            clear_display()
            last_image_filename = None
        except Exception as e:
            print(f"Error deleting image: {e}")

# Tkinter GUI setup
root = tk.Tk()
root.title("QC Image Annotation")

root.attributes("-topmost", True)
root.state('zoomed')
# root.minsize(600,450)

# Dropdown menu for annotations
annotation_var = tk.StringVar(value=annotations[0])
annotation_menu = ttk.Combobox(root, textvariable=annotation_var, values=annotations, state="readonly", font=("Arial", 14), width=25)
annotation_menu.pack(pady=10)

# Take Picture button
take_picture_button = tk.Button(root, text="Take Picture", command=on_take_picture, font=("Arial", 16), width=15, height=2)
take_picture_button.pack(pady=10)

# Reject Picture button
reject_picture_button = tk.Button(root, text="Reject Picture", command=on_reject_picture, font=("Arial", 16), width=15, height=2)
reject_picture_button.pack(pady=10)

# Display area for the captured image
display_label = tk.Label(root)
display_label.pack(pady=10)

# Label to show success or delete message
success_message = tk.Label(root, text="", font=("Arial", 12), fg="green")
success_message.pack(pady=10)

# Run the GUI loop
root.mainloop()

# Release the camera when the application is closed
if camera:
    camera.StopGrabbing()
    camera.Close()

main_directory = "annotated_images"
process_main_directory(main_directory)
