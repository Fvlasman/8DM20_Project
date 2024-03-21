# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:36:57 2024

@author: 20192547
"""
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import RandZoom,RandRotate,Rand3DElastic
import os


DATA_DIR = Path.cwd().parent / "TrainingData"

patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]

RandZoomFilt = RandZoom(prob = 1, min_zoom = 0.8, max_zoom = 1.2)
RandRotateFilt = RandRotate(range_x=0.3, range_y=0.3, range_z=0.3, prob=1)
Rand3DElasticFilt = Rand3DElastic(sigma_range = (7,7,7), magnitude_range = (50,50,50) , prob=1, padding_mode = "zeros")

def save_AUG_image_as_mhd(image_array, filename,target_path):

    if not filename.endswith('.mhd'):
        filename += '.mhd'
        
    save_path = os.path.join(target_path, filename)
    
    image = sitk.GetImageFromArray(image_array)
    
    sitk.WriteImage(image, save_path)
    
    return save_path

target_path = Path.cwd().parent / "Aug_images"

for path in range(len(patients)):
    f_image_array =sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patients[path], "prostaat.mhd")))
    augmented_image=RandZoomFilt(RandRotateFilt(Rand3DElasticFilt(f_image_array)))
    #plot
    name= f"augmented_image{path}.mhd"  
    save_AUG_image_as_mhd(augmented_image,name,target_path)
    