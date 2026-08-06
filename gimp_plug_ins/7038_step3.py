# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 07:42:07 2024

@author: Ryan.Larson
"""

from gimpfu import *

def process_7038step3(image, drawable):
    ###########################################################
    # PERFORM THRESHOLDING BEFORE THIS STEP!!!!!
    ###########################################################
    
    # Set the active layer as the top layer
    pdb.gimp_image_set_active_layer(image, image.layers[0])
    
    # Remove the bottom two layers
    pdb.gimp_image_remove_layer(image, image.layers[7])
    pdb.gimp_image_remove_layer(image, image.layers[6])
    
    # Set the range value to the number of rectangle layers
    for i in range(6):
        pdb.gimp_layer_set_opacity(image.layers[i], 100)
    
    # pdb.gimp_context_set_foreground((0, 0, 0))
    black_layer = pdb.gimp_layer_new(image, image.width, image.height, GRAY_IMAGE, "Black Layer", 100, NORMAL_MODE)
    pdb.gimp_image_insert_layer(image, black_layer, None, len(image.layers))
    
    # Merge down until a single layer remains
    while len(image.layers) > 1:
        pdb.gimp_image_merge_down(image, image.layers[0], EXPAND_AS_NECESSARY)
        
    # Final threshold operation
    drawable = image.active_layer
    pdb.gimp_drawable_threshold(drawable, 0, 1, 1)

    # Step 5: Get filename, set output name, and save as PNG
    original_filename = pdb.gimp_image_get_filename(image)
    output_name = original_filename.replace(".jpg", "_THERMAL_SHOCK.png")

    drawable = pdb.gimp_image_get_active_drawable(image)
    pdb.file_png_save(image, drawable, output_name, output_name, 0, 9, 1, 1, 1, 1, 1)

    gimp.displays_flush()

def register_script():
    register(
        "7038_step3",
        "Process all open images",
        "A script to process all open images in GIMP",
        "Ryan Larson",
        "Your License",
        "2024",
        "<Image>/Filters/Custom/7038 Step 3",
        "*",  # Specify acceptable image types
        [],
        [],
        process_7038step3)

register_script()
main()
