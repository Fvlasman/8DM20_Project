# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:02:43 2024

@author: 20192832
"""

from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import SimpleITK as sitk

#Read image masks from the patients, mhd files    
atlas_1 = sitk.ReadImage("TrainingData-GroupAssignment\TrainingData\p102\prostaat.mhd")    
atlas_2 = sitk.ReadImage("TrainingData-GroupAssignment\TrainingData\p108\prostaat.mhd")    
atlas_3 = sitk.ReadImage("TrainingData-GroupAssignment\TrainingData\p116\prostaat.mhd")
atlas_4 = sitk.ReadImage("TrainingData-GroupAssignment\TrainingData\p128\prostaat.mhd")
atlas_5 = sitk.ReadImage("TrainingData-GroupAssignment\TrainingData\p129\prostaat.mhd")

#Convert image format to a 3 dimensional array to compare the images
image_1 = sitk.GetArrayViewFromImage(atlas_1)
image_2 = sitk.GetArrayViewFromImage(atlas_2)
image_3 = sitk.GetArrayViewFromImage(atlas_3)
image_4 = sitk.GetArrayViewFromImage(atlas_4)
image_5 = sitk.GetArrayViewFromImage(atlas_5)

#Obtaining the 3 dimensional shape of the images
image_shape = image_1.shape

#Creating an empty array which intializes our new atlas image
combined_atlas = np.zeros((image_shape[0], image_shape[1], image_shape[2]), dtype=int)
vote_list = np.zeros(5, dtype = int)

#Looking at every pixel based on the pixel values for each images, assign new value.
#New value of the atlas will be 1 if three or more images have pixel value of 1.
for i in range(0, image_shape[0]):
    for j in range(0, image_shape[1]):
        for k in range(0, image_shape[2]):
            if image_1[i, j, k] == 1:
                vote_list[0] =1
            if image_2[i, j, k] == 1:
                vote_list[1] =1
            if image_3[i, j, k] == 1:
                vote_list[2] =1
            if image_4[i, j, k] == 1:
                vote_list[3] =1
            if image_5[i, j, k] == 1:
                vote_list[4] =1
            if sum(vote_list) >= 3:
                combined_atlas[i,j,k] = 1
            vote_list = np.zeros(5, dtype = int)

# Display the image slice from the middle of the stack, z axis
z = int(atlas_1.GetDepth()/2)
npa_zslice = sitk.GetArrayViewFromImage(atlas_1)[z,:,:]
npa_zslice_2 = sitk.GetArrayViewFromImage(atlas_2)[z,:,:]
npa_zslice_3 = sitk.GetArrayViewFromImage(atlas_3)[z,:,:]
npa_zslice_4 = sitk.GetArrayViewFromImage(atlas_4)[z,:,:]
npa_zslice_5 = sitk.GetArrayViewFromImage(atlas_5)[z,:,:]
npa_zslice_atlas = combined_atlas[z, :, :]

# Six plots displaying the binary mask displaying the prostate. The sixth image
# displays the atlas created by majority voting of 5 prostate images.
fig = plt.figure(figsize=(10,3))

fig.add_subplot(2,3,1)
plt.imshow(npa_zslice)
plt.title('Segmentation of P102', fontsize=10)
plt.axis('off')

fig.add_subplot(2,3,2)
plt.imshow(npa_zslice_2)    
plt.title('Segmentation of P108', fontsize=10)
plt.axis('off')

fig.add_subplot(2,3,3)
plt.imshow(npa_zslice_3)
plt.title('Segmentation of p116', fontsize=10)
plt.axis('off')

fig.add_subplot(2,3,4)
plt.imshow(npa_zslice_4)
plt.title('Segmentation of p128', fontsize=10)
plt.axis('off')

fig.add_subplot(2,3,5)
plt.imshow(npa_zslice_5)
plt.title('Prostate segmentation of p129', fontsize=10)
plt.axis('off')

fig.add_subplot(2,3,6)
plt.imshow(npa_zslice_atlas)
plt.title('Combined Segmentation by Majority Vote', fontsize=10)
plt.axis('off')

#Writing the obtained atlas into an MHD file
atlas_mhd = sitk.GetImageFromArray(combined_atlas)
sitk.WriteImage(atlas_mhd, "average_atlas_v2.mhd")
