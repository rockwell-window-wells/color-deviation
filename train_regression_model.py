# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:46:41 2024

@author: Ryan.Larson
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, "Raw image file"]
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Convert targets to tensor
        # targets = self.dataframe.iloc[idx, 1:].values.astype(float)
        # selected_cols = ["% pixels in thermal shock"]
        # selected_cols = ["Base Color Avg Delta E", "Thermal Shock Avg Delta E", "% pixels in thermal shock"]
        selected_cols = ["Max Delta E - Base to Shock"]
        targets = self.dataframe.loc[idx, selected_cols].values.astype(float)
        targets = torch.tensor(targets, dtype=torch.float32)
        
        return image, targets

# Define CNN model (replace with a more complex architecture as needed)
class RegressionCNN(nn.Module):
    def __init__(self, num_outputs):
        super(RegressionCNN, self).__init__()
        # Use pretrained model (e.g., ResNet18), replace last layer
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)

    def forward(self, x):
        return self.model(x)

# Data preparation
def prepare_data(dataframe, image_dir, batch_size=32, test_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(dataframe, image_dir, transform)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Main function
def main():
    # Load the dataframe
    df = pd.read_csv("C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/training_data.csv")  # Update with your actual file path
    image_dir = "C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/training_images"  # Update with your image directory

    # Parameters
    num_outputs = 1  # Number of target variables (delta_E and pct_coverage)
    num_epochs = 40
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    train_loader, test_loader = prepare_data(df, image_dir, batch_size)

    # Model, criterion, optimizer
    model = RegressionCNN(num_outputs).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "regression_model.pth")

if __name__ == "__main__":
    main()
