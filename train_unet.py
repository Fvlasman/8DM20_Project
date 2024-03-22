import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

if torch.cuda.is_available():
    print("CUDA is available. GPU is being used.")
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. GPU is not being used.")


# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-4
TOLERANCE = 0.03  # for early stopping

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

# initialise model, optimiser, and loss function
loss_function = utils.DiceBCELoss() # import custom loss function from utils module 
unet_model = u_net.UNet(num_classes=1).to(device) # import unet implementation from unet module / num_classes is number of output classes of the segmentation
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE) # use a default pytorch optimiser

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # training iterations
    # tqdm is for timing iteratiions
    # the required implementation of training iterations in pytorch is composed of 5 steps:
    for inputs, labels in tqdm(dataloader, position=0):
        optimizer.zero_grad() # (1) zeroing the gradients in each iterations
        outputs = unet_model(inputs.to(device))  # (2) forward pass of the model
        loss = loss_function(outputs, labels.float().to(device)) # (3) computing the loss
        loss.backward()  # (4) backpropagating the loss
        current_train_loss += loss.item()
        optimizer.step()  # (5) stepping the optimiser (update the weights)

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval() # turns off the training setting to allow evaluation 
        # evaluation validation loss
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs = unet_model(inputs.to(device))  # (2) forward pass of the model
            loss = loss_function(outputs, labels.float().to(device)) # (3) computing the loss
            current_valid_loss += loss.item()

        unet_model.train() # turns training setting back on

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        weights_dict = {k: v.cpu() for k, v in unet_model.state_dict().items()}
        if epoch > 9:
            torch.save(
                weights_dict,
                CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
            )
