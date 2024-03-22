import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import u_net
import utils

import SimpleITK as sitk
import os

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to stored training checkpoints
DATA_DIR = Path.cwd() / "TrainingData"
BEST_EPOCH = 99
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights" / f"u_net_{BEST_EPOCH}.pth"

# hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)

unet_model = u_net.UNet(num_classes=1)
unet_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
unet_model.eval()

# TODO
# apply for all images and compute Dice score with ground-truth.
# output .mhd images with the predicted segmentations
with torch.no_grad():
    predict_index = 75  # here I just chose a random slice for testing
    # you should do this for all slices
    (input, target) = valid_dataset[predict_index]
    output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
    prediction = torch.round(output)

    plt.subplot(131)
    plt.imshow(input[0], cmap="gray")
    plt.subplot(132)
    plt.imshow(target[0])
    plt.subplot(133)
    plt.imshow(prediction[0, 0])
    plt.show()

predictions = []
for i in range(len(valid_dataset)):
    with torch.no_grad():
        input, _ = valid_dataset[i]
        output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
        prediction = torch.round(output)
        predictions.append(prediction.squeeze().numpy())  # 将预测添加到列表中


def dice_coefficient(true_mask, pred_mask):
    smooth = 1.0
    true_mask = true_mask.flatten()
    pred_mask = pred_mask.flatten()
    intersection = (true_mask * pred_mask).sum()
    return (2. * intersection + smooth) / (true_mask.sum() + pred_mask.sum() + smooth)


dice_scores = []
for i in range(len(valid_dataset)):
    with torch.no_grad():
        input, target = valid_dataset[i]
        output = torch.sigmoid(unet_model(input[np.newaxis, ...]))
        prediction = torch.round(output)
        dice_score = dice_coefficient(target.numpy(), prediction.squeeze().numpy())
        dice_scores.append(dice_score)


average_dice_score = sum(dice_scores) / len(dice_scores)
print(f"Average Dice Score: {average_dice_score}")


target_path = "D:\\uu\\capita_unet\\seg_result"

os.makedirs(target_path, exist_ok=True)


def save_prediction_as_mhd(image_array, filename):
    if not filename.endswith('.mhd'):
        filename += '.mhd'

    base_dir = "D:\\uu\\capita_unet\\seg_result"

    save_path = os.path.join(base_dir, filename)

    image = sitk.GetImageFromArray(image_array)

    sitk.WriteImage(image, save_path)

    return save_path


for i, prediction in enumerate(predictions):
    filename = f"prediction_{i}.mhd"
    save_path = save_prediction_as_mhd(prediction, filename)