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

if os.path.exists('results') is False:
    os.mkdir('results')

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

transform_path = os.path.join('results', 'TransformParameters.0.txt')
result_path = os.path.join('results', 'result.0.tiff')

direc = 'p102'

el.register(
    fixed_image=os.path.join(one_patient_path, 'mr_bffe.mhd'),
    moving_image=os.path.join(atlas_path, 'atlas_mr_bffe_2_affine.mhd'),
    parameters=[os.path.join(parameter_path, 'parameterswithpenalty_register_pat.txt')],
    output_dir='results')
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)
# Transform a new image with the transformation parameters
tr.transform_image(os.path.join(atlas_path, 'atlas_prostaat_2_affine.mhd'), output_dir=r'results')
predicted_prostate = sitk.ReadImage(os.path.join(r'C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - '
                                                 r'Capita selecta\python\results', 'result.mhd'))
predicted_prostate_array = sitk.GetArrayFromImage(predicted_prostate)

atlas = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(atlas_path, 'atlas_mr_bffe_2_affine.mhd')))
original_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(one_patient_path, 'prostaat.mhd')))
registered_moving_image = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.0.mhd"))
moving_prostate = sitk.GetArrayFromImage(sitk.ReadImage(r"C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - Capita selecta\python\results\result.mhd"))

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(atlas[40,:,:], cmap='gray')
ax[0].set_title('Atlas')
ax[1].imshow(original_image[40,:,:], cmap='gray')
ax[1].set_title('Orginial image')
ax[2].imshow(registered_moving_image[40,:,:], cmap='gray')
ax[2].set_title('Registered image')
ax[3].imshow(moving_prostate[40,:,:], cmap='gray')
ax[3].set_title('Moving prostate')
[x.set_axis_off() for x in ax]
plt.show()
