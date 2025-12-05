import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.io import decode_image
import matplotlib.pyplot as plt
from cnn_architectures.bear_net import BearNet


# Define custom dataset
class BearCarDataset(Dataset):
    """
    Customized dataset
    """

    def __init__(self, labels_file_path, img_dir):
        self.labels_df = pd.read_csv(labels_file_path, header=None)
        self.img_dir = img_dir
        self.trans_in = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = self.trans_in(decode_image(img_path))
        target_steering = torch.as_tensor(
            self.labels_df.iloc[idx, 1], dtype=torch.float32
        )
        target_throttle = torch.as_tensor(
            self.labels_df.iloc[idx, 2], dtype=torch.float32
        )
        return image, target_steering, target_throttle


# SETUP
# Pass in data directory name as CLI argument
# e.g. python train.py 2025-12-13-14-15
if len(sys.argv) != 2:
    print("Training script needs data! Please provide date and time!")
    sys.exit(1)  # exit with an error code
else:
    data_datetime = sys.argv[1]
# Designate processing unit for CNN training
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# Define training process
def train(dataloader, model, loss_fn, optimizer):
    """
    training function for 1 epoch
    """
    model.train()
    num_used_samples = 0
    ep_loss = 0.0
    for i, (img, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = img.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples += target.shape[0]
        print(
            f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]"
        )
        ep_loss = (ep_loss * i + batch_loss.item()) / (i + 1)
    return ep_loss


# Define validation process
def validate(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.0
    with torch.no_grad():
        for i, (img, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target = img.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)
            ep_loss = (ep_loss * i + batch_loss.item()) / (i + 1)
    return ep_loss


# Instantiate dataset
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    data_datetime,
)
raw_labels_file_path = os.path.join(data_dir, "labels.csv")  # the name of the csv file
# Split train/val (90:10)
raw_labels_df = pd.read_csv(
    raw_labels_file_path,
    header=None,
    names=["image_id", "steering_value", "throttle_value"],
)
val_ids = np.arange(
    int(len(raw_labels_df) / 2) - int(len(raw_labels_df) * 0.1),  # lower bound
    int(len(raw_labels_df) / 2) + int(len(raw_labels_df) * 0.1),  # upper bound
)
val_df = raw_labels_df.iloc[val_ids]
train_df = raw_labels_df.drop(val_ids)
val_df = val_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
print(f"train size: {len(train_df)}, validation size: {len(val_df)}")
# Generate train/val annotation files
train_df.to_csv(
    os.path.join(data_dir, "labels_train.csv"),
    index=False,
    header=False,
)
val_df.to_csv(
    os.path.join(data_dir, "labels_val.csv"),
    index=False,
    header=False,
)
# Create dataset and dataloader
img_dir = os.path.join(data_dir, "images")
train_set = BearCarDataset(os.path.join(data_dir, "labels_train.csv"), img_dir)
val_set = BearCarDataset(os.path.join(data_dir, "labels_val.csv"), img_dir)
train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=128)
# Create directory for saving trained models
model_dir = os.path.join(data_dir, "models")
if not os.path.exists(model_dir):
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# Instantiate model and config training
model = BearNet().to(DEVICE)  # choose the architecture class from cnn_network.py
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
max_epochs = 64
patience = 5
best_loss = float("inf")  # best loss on validation data
since_best_counter = 0
train_losses = []
val_losses = []


# LOOP
# Optimize model
for ep in range(max_epochs):
    print(f"Epoch {ep + 1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_val_loss = validate(val_dataloader, model, loss_fn)
    print(
        f"Epoch {ep + 1} training loss: {ep_train_loss}, validation loss: {ep_val_loss}"
    )
    train_losses.append(ep_train_loss)
    val_losses.append(ep_val_loss)
    model_name = f"ep{ep + 1}-mse{ep_val_loss:.4f}"
    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}.pth"))
    # Early stopping
    if ep_val_loss < best_loss:
        best_loss = ep_val_loss
        since_best_counter = 0  # Reset counter if validation loss improved
        try:
            os.remove(os.path.join(data_dir, "best_model.pth"))
            print("Last best model file has been deleted successfully.")
        except FileNotFoundError:
            print(f"File '{os.path.join(data_dir, 'best_model.pth')}' not found.")
        except Exception as e:
            print(f"Error occurred while deleting the file: {e}")
        torch.save(model.state_dict(), os.path.join(data_dir, "best_model.pth"))
        print(
            f"Best model: {model_name} saved at {os.path.join(data_dir, 'best_model.pth')}"
        )
    else:
        since_best_counter += 1
        print(f"{since_best_counter} epochs since best model")
        if since_best_counter >= patience:
            print("Early stopping triggered!")
            break
print("Optimize Done!")  # debug
# Graph training process
plt.plot(range(len(train_losses)), train_losses, "b--", label="Training")
plt.plot(range(len(val_losses)), val_losses, "orange", label="Validation")
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title(model_name)
plt.savefig(os.path.join(model_dir, "learning_curve.png"))
