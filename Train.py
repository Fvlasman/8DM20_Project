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
test_path = os.path.join(atlas_path, 'atlas_mr_bffe.mhd')
print(os.path.join(atlas_path, "atlas_prostaat.mhd"))
atlas_image = sitk.ReadImage(os.path.join(atlas_path, "atlas_prostaat_7.mhd"))
image_array = sitk.GetArrayFromImage(atlas_image)

# plt.imshow(image_array[:, :, 2], cmap='gray')
# plt.show()

if os.path.exists('results') is False:
    os.mkdir('results')

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

transform_path = os.path.join('results', 'TransformParameters.0.txt')
result_path = os.path.join('results', 'result.0.tiff')
scores = []

directories = os.listdir(training_path)

for dir in directories[2:]:
    dir_path = os.path.join(training_path, dir)
    moving_image = sitk.ReadImage(os.path.join(dir_path, 'mr_bffe.mhd'))
    el.register(
        fixed_image=os.path.join(dir_path, 'mr_bffe.mhd'),
        moving_image=os.path.join(atlas_path, 'atlas_mr_bffe_7.mhd'),
        parameters=[os.path.join(parameter_path, 'parameterswithpenalty.txt')],
        output_dir='results')
    tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=TRANSFORMIX_PATH)
    # Transform a new image with the transformation parameters
    tr.transform_image(os.path.join(atlas_path, 'atlas_prostaat_7.mhd'), output_dir=r'results')
    predicted_prostate = sitk.ReadImage(os.path.join(r'C:\Users\fvlas\OneDrive - TU Eindhoven\vakken TUE\8DM20 - '
                                                     r'Capita selecta\python\results', 'result.mhd'))
    predicted_prostate_array = sitk.GetArrayFromImage(predicted_prostate)
    # calculate DICE scores
    actual_prostate = sitk.ReadImage(os.path.join(dir_path, 'prostaat.mhd'))
    actual_prostate_array = sitk.GetArrayFromImage(actual_prostate)
    common_array = predicted_prostate_array * actual_prostate_array
    dice = (2*np.sum(common_array))/(np.sum(predicted_prostate_array)+np.sum(actual_prostate_array))
    scores.append(dice)

print(scores)
test = 0
