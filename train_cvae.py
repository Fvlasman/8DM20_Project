# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:04:48 2024

@author: 20192547
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
import cvae
import os
import SimpleITK as sitk

# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd().parent / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "vae_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 2
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 256

# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


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

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
cvae_model = cvae.CVAE().to(device)# TODO 
optimizer = torch.optim.Adam(cvae_model.parameters(),lr=LEARNING_RATE)# TODO 
# add a learning rate scheduler based on the lr_lambda function
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)# TODO ??

train_losses = []
valid_losses = []
# training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # TODO 
    # training iterations
    for inputs, seg_labels  in dataloader:
        optimizer.zero_grad()
        
        output, mu, logvar=cvae_model(inputs.to(device),seg_labels.to(device))
        #cvae_loss(inputs, recons, mu, logvar, seg_labels)
        loss=cvae.cvae_loss(inputs.to(device),output.to(device), mu, logvar,seg_labels.to(device))
        loss.backward()
        optimizer.step()
        current_train_loss += loss.item()

    # evaluate validation loss
    with torch.no_grad():
        cvae_model.eval()
        # TODO
        for inputs, seg_labels in valid_dataloader:
            output, mu, logvar=cvae_model(inputs.to(device),seg_labels.to(device))
            loss=cvae.cvae_loss(inputs.to(device),output.to(device), mu, logvar, seg_labels.to(device))
            current_valid_loss += loss.item()
        cvae_model.train()
        
        
    # Calculate average training and validation losses
    avg_train_loss = current_train_loss / len(dataloader)
    avg_valid_loss = current_valid_loss / len(valid_dataloader)

    # Append losses to lists
    train_losses.append(avg_train_loss)
    valid_losses.append(avg_valid_loss)
    
    
    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )
    scheduler.step() # step the learning step scheduler

    # save examples of real/fake images
    if (epoch + 1) % DISPLAY_FREQ == 0:
        img_grid = make_grid(
            torch.cat((output[:5], inputs[:5])), nrow=5, padding=12, pad_value=-1
        ) #x_recon x_real
        writer.add_image(
            "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
        )
    print("\tEpoch", epoch + 1, "\tTraining Loss: ", current_train_loss)

# Plot learning curves
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

torch.save(cvae_model.state_dict(), CHECKPOINTS_DIR / "cvae_model2epochs.pth")

#%%
#Function to generate prostate images with the cvae
def generate_image(noise, aug_seg, cvae_model):
    with torch.no_grad():
        cvae_model.eval()
        #fake_images, mu, logvar=vae_model(noise)
        gen = cvae.Generator()
        fake_images=gen(noise,aug_seg.to(device))#have to adjust these segmentations
        plt.figure()
        plt.imshow(fake_images[0,0,:,:],cmap = "gray")
        plt.show()
    return fake_images
#Function tosave the generate prostate images
def save_GEN_image_as_mhd(image_array, filename,target_path):

    if not filename.endswith('.mhd'):
        filename += '.mhd'
        
    save_path = os.path.join(target_path, filename)
    
    image = sitk.GetImageFromArray(image_array)
    
    sitk.WriteImage(image, save_path)
    
    return save_path

# load the data augmentated segmentation images
DATA_DIR_AUG = Path.cwd().parent / "Aug_images"

augmented_images_paths = [
    path
    for path in DATA_DIR_AUG.iterdir()
    if not any(part.startswith(".") for part in path.parts)
]
mhd_files = [filename for filename in augmented_images_paths if filename.suffix.lower() == ".mhd"]

dataset2 = utils.GenMRDataset(mhd_files, IMAGE_SIZE)
dataloader2 = DataLoader(
    dataset2,
    batch_size=len(dataset2),
    shuffle=False,
    drop_last=True,
    pin_memory=True,
)

#make the noise and generate the images and save them
noise = cvae.get_noise(len(dataset2), Z_DIM, device=device)
count=0
target_path = Path.cwd().parent / "Gen_images"
for seg_label in dataloader2:
    fake_image=generate_image(noise[count], seg_label, cvae_model)
    name= f"gen_image{count}.mhd"
    save_GEN_image_as_mhd(fake_image,name,target_path)
    count=count+1




