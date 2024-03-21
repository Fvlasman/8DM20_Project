from __future__ import print_function, absolute_import
import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import SimpleITK as sitk
import time

elastics_dir = r'C:\Users\fvlas\OneDrive\Documents\elastix\elastix-5.0.0-win64'
ELASTIX_PATH = os.path.join(elastics_dir, r'elastix.exe')
TRANSFORMIX_PATH = os.path.join(elastics_dir, r'transformix.exe')
parameter_path = (r'C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita '
                  r'selecta\ImagesforPractical\ImagesforPractical\chest_xrays')

atlas_path = r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\TrainingData\atlas"
training_path = r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\TrainingData"

one_patient_path = r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\TrainingData\p119"
second_patient_path = r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\TrainingData\p128"

results_path = r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results"

if os.path.exists('results') is False:
    os.mkdir('results')

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

transform_path = os.path.join('results', 'TransformParameters.0.txt')

direc = 'p102'

el.register(
    fixed_image=os.path.join(one_patient_path, 'mr_bffe.mhd'),
    moving_image=os.path.join(atlas_path, 'atlas_mr_bffe_rigid_108-116-128.mhd'),
    parameters=[os.path.join(parameter_path, 'parameterswithpenalty.txt')],
    output_dir='results')
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

# Transform a new image with the transformation parameters
tr.transform_image(os.path.join(atlas_path, 'atlas_prostaat_rigid_108-116-128.mhd'), output_dir=r'results')

atlas = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(atlas_path, 'atlas_mr_bffe_rigid_108-116-128.mhd')))
atlas_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(atlas_path, 'atlas_prostaat_rigid_108-116-128.mhd')))
original_prostate = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(one_patient_path, 'prostaat.mhd')))
original_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(one_patient_path, 'mr_bffe.mhd')))
registered_moving_image_rigid = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.0.mhd"))
moving_prostate_rigid = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.mhd"))

# Do the second registration (bspline)

el.register(
    fixed_image=os.path.join(one_patient_path, 'mr_bffe.mhd'),
    moving_image=os.path.join(results_path, 'result.0.mhd'),
    parameters=[os.path.join(parameter_path, 'parameterswithpenalty_bspline.txt')],
    output_dir='results')

tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

# Transform a new image with the transformation parameters
tr.transform_image(os.path.join(results_path, 'result.mhd'), output_dir=r'results')

registered_moving_image_bspline = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.0.mhd"))
moving_prostate_bspline = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.mhd"))
moving_prostate_bspline[moving_prostate_bspline > 0.1] = 1
moving_prostate_bspline[moving_prostate_bspline <= 0.1] = 0

fig, ax = plt.subplots(2, 4, figsize=(20, 5))
test = ax[0][0]
ax[0][0].imshow(original_image[40,:,:], cmap='gray')
ax[0][0].set_title('Original image')
ax[0][1].imshow(atlas[40,:,:], cmap='gray')
ax[0][1].set_title('moving image')
ax[0][2].imshow(registered_moving_image_rigid[40,:,:], cmap='gray')
ax[0][2].set_title('Registered image after rigid')
ax[0][3].imshow(registered_moving_image_bspline[40,:,:], cmap='gray')
ax[0][3].set_title('Moving image after bspline')
ax[1][0].imshow(original_prostate[40,:,:], cmap='gray')
ax[1][0].set_title('original prostate')
ax[1][1].imshow(atlas_seg[40,:,:], cmap='gray')
ax[1][1].set_title('atlas seg')
ax[1][2].imshow(moving_prostate_rigid[40,:,:], cmap='gray')
ax[1][2].set_title('Moving prostate after rigid')
ax[1][3].imshow(moving_prostate_bspline[40,:,:], cmap='gray')
ax[1][3].set_title('Moving prostate after bspline')
for row in ax:
    for x in row:
        x.set_axis_off()

plt.show()
