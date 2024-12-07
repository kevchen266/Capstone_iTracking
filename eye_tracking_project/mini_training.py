from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torchvision import transforms
from .queue_manager import dataloader_q
from PIL import Image # type: ignore
from .CNN_model.Gazetrack import Gazetrack
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
dataloader_q contain [[left_eye1,right_eye1, cropped_coordinates1], [left_eye2,right_eye2, cropped_coordinates2]........]

-> transform
-> dataloader
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gazetrack().to(device)
criterion = nn.SmoothL1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
# Paths for weights
default_model_path = os.path.join(BASE_DIR, "CNN_model/cnn_model_weights.pth")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class MiniTrainDataset(Dataset):
    """
    load data from dataloader_q and transform into dataloader
    """

    def __init__(self, dataloader_q, transform=None):
        self.data = list(dataloader_q)  # Convert the deque to a list
        self.transform = transform

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data at the given index
        left_eye, right_eye, coordinates = self.data[idx]

        # Apply transformations to the images
        if self.transform:
            left_eye_tensor = self.transform(Image.fromarray(left_eye))
            right_eye_tensor = self.transform(Image.fromarray(right_eye))
        else:
            left_eye_tensor = torch.tensor(left_eye, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
            right_eye_tensor = torch.tensor(right_eye, dtype=torch.float32).permute(2, 0, 1)

        # Normalize gaze coordinates to a tensor
        normalized_gaze_tensor = torch.tensor(coordinates, dtype=torch.float32)

        return left_eye_tensor, right_eye_tensor, normalized_gaze_tensor


def mini_train_step(model, optimizer, criterion, device, epochs=1):
    """Perform a mini-training step with the collected calibration data.
        *save the endpoint as updated_cnn_model_weight.pth for preadiction phase.
    """
    

    model.train()
    dataset = MiniTrainDataset(dataloader_q, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("START TRAINING")

    for epoch in range(epochs):
        running_loss = 0.0
        print("START TRAINING")
        for left_eye, right_eye, label in dataloader:
            print(f"Data Shapes -> Left Eye: {left_eye.shape}, Right Eye: {right_eye.shape}, Label: {label.shape}")

            # Move tensors to the appropriate device
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            label = label.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(left_eye, right_eye)
            print(f"Model Outputs: {outputs}, Labels: {label}")

            # Compute loss and backpropagation
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
    updated_model_path = os.path.join(BASE_DIR, "CNN_model/update_cnn_model_weights.pth")
    torch.save(model.state_dict(), updated_model_path)

    
  


