import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# List of classification annotations
annotations = [
    "6024 PASS", "6024 FAIL", "6038 PASS", "6038 FAIL",
    "7024 PASS", "7024 FAIL", "7038 PASS", "7038 FAIL"
]

# Directory to save images
output_dir = "annotated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize image counter
image_counter = 0

# Open the camera once and keep it open
cap = cv2.VideoCapture(0)

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
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        return None
    
    return frame

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
    img = img.resize((400, 300))  # Resize for display
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
cap.release()
