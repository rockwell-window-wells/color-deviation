# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:00:35 2024

@author: Ryan.Larson

General script for training 
"""

import cv2
from tkinter import Tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models.detection as detection
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim



def get_images_directory():
    root = Tk()
    root.wm_attributes('-topmost', 1)
    image_dir = fd.askdirectory(
                parent=root,
                title="Select images directory",
                initialdir='C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/test_training/'
                )
    root.destroy()
    return image_dir
    
if __name__ == "__main__":
    image_dir = get_images_directory()