# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 07:42:07 2024

@author: Ryan.Larson
"""

import json
import os
from gimpfu import *

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_layer_by_name(image, layer_name):
    for layer in image.layers:
        if layer.name == layer_name:
            return layer
    return None  # Return None if no layer with the given name is found

def add_reference_mask(image, filename):
    # pdb.gimp_image_undo_group_start(image)
    layer = pdb.gimp_file_load_layer(image, filename)
    pdb.gimp_image_add_layer(image, layer, 0)
    # pdb.gimp_image_undo_group_end(image)
    
def invert_and_subtract_mask(image):
    pdb.gimp_drawable_invert(image.active_layer, 0)
    pdb.gimp_layer_set_mode(image.active_layer, SUBTRACT_MODE)
    pdb.gimp_image_merge_down(image, image.active_layer, 2)
    
def blacken_sides(image):
    layer = image.active_layer
    
    # Create a rectangular selection (x, y are the top-left corner coordinates)
    x = 0
    y = 0
    width = 450
    height = layer.height
    pdb.gimp_image_select_rectangle(image, CHANNEL_OP_REPLACE, x, y, width, height)
    
    # Set the foreground color to black
    pdb.gimp_context_set_foreground((0, 0, 0))  # RGB for black

    # Fill the selected area with the foreground color (black)
    pdb.gimp_edit_fill(layer, FOREGROUND_FILL)

    # Deselect everything
    pdb.gimp_selection_none(image)
    
    x = 3250
    y = 0
    width = layer.width - x
    height = layer.height
    
    pdb.gimp_image_select_rectangle(image, CHANNEL_OP_REPLACE, x, y, width, height)
    
    # Set the foreground color to black
    pdb.gimp_context_set_foreground((0, 0, 0))  # RGB for black

    # Fill the selected area with the foreground color (black)
    pdb.gimp_edit_fill(layer, FOREGROUND_FILL)

    # Deselect everything
    pdb.gimp_selection_none(image)
    
def separate_faces(image):
    # 7038 face selection rectangles
    rects = [(450, 0, 2800, 150),
             (450, 244, 2800, 276),
             (450, 672, 2800, 283),
             (450, 1103, 2800, 328),
             (450, 1513, 2800, 394),
             (450, 1922, 2800, 238),
             ]
    
    # pdb.gimp_image_undo_group_start(image)
    
    thermal_shock_layer = get_layer_by_name(image, "Thermal Shock")
    
    for rect in rects:
        x, y, width, height = rect
        
        # Select the rectangular area
        pdb.gimp_image_select_rectangle(image, CHANNEL_OP_REPLACE, x, y, width, height)
        
        # Duplicate the active layer, crop to selection, and add it to the image
        copy_layer = pdb.gimp_layer_copy(thermal_shock_layer, True)
        pdb.gimp_image_insert_layer(image, copy_layer, None, 0)
        # pdb.gimp_layer_resize_to_image_size(copy_layer)
        
        # Clear areas outside the selection
        pdb.gimp_selection_invert(image)  # Invert the selection to select everything outside the rectangle
        pdb.gimp_edit_clear(copy_layer)   # Clear outside the selection
        pdb.gimp_selection_none(image)     # Deselect everything
        
        # Set the opacity of the copied layer to 3%
        pdb.gimp_layer_set_opacity(copy_layer, 3)
    
    pdb.gimp_selection_none(image)
    
    pdb.gimp_layer_set_opacity(thermal_shock_layer, 0)
    
    # pdb.gimp_image_undo_group_end(image)
    
    

def process_7038step2(image, drawable):
    ######################################
    # Manually Use Difference of Gaussians
    ######################################
    
    # Overlay the appropriate reference mask and subtract everything unnecessary
    # filename = "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/masks/reference_masks/7038_ref_mask.png"
    config = load_config()
    filename = config.get("7038_ref_mask")
    add_reference_mask(image, filename)
    invert_and_subtract_mask(image)
    
    blacken_sides(image)
    
    separate_faces(image)
    
    gimp.displays_flush()

def register_script():
    register(
        "7038_step2",
        "Thermal shock 7038 step 2",
        "After Difference of Gaussians, masks the image and blackens the sides, then separates faces for thresholding",
        "Ryan Larson",
        "GNU GPLv3",
        "2024",
        "<Image>/Filters/Custom/7038 Step 2",
        "*",  # Specify acceptable image types
        [],
        [],
        process_7038step2)

register_script()
main()
