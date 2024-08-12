import os
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.models.detection as detection
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Base directory paths
annotations_dir = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/test_training/annotations/'
images_dir = 'C:/Users/Ryan.Larson.ROCKWELLINC/github/color-deviation/test_training/'

# Prepare the data
data = []
for filename in os.listdir(annotations_dir):
    if filename.endswith('.json'):
        with open(os.path.join(annotations_dir, filename), 'r') as f:
            annotation = json.load(f)
            relative_image_path = os.path.join(os.path.dirname(os.path.join(annotations_dir, filename)), annotation['imagePath'])
            image_path = os.path.abspath(relative_image_path)
            if os.path.exists(image_path):
                points = annotation['shapes'][0]['points']
                xmin = int(min(points[0][0], points[1][0]))
                ymin = int(min(points[0][1], points[1][1]))
                xmax = int(max(points[0][0], points[1][0]))
                ymax = int(max(points[0][1], points[1][1]))
                data.append((image_path, [xmin, ymin, xmax, ymax], 1))
            else:
                print(f"Warning: {image_path} not found and will be skipped.")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Custom dataset
class DenaliDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, bbox, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = {}
        target['boxes'] = torch.tensor([bbox], dtype=torch.float32)
        target['labels'] = torch.tensor([label], dtype=torch.int64)
        return image, target

# Create dataset
dataset = DenaliDataset(data, transform=transform)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Faster R-CNN model
model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# model = detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

# Replace the classifier with a new one for our specific class (including background as class 0)
num_classes = 2  # 1 class (denali6038) + background
# in_features = model.roi_heads.box_predictor.in_features
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to the correct device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {losses.item():.4f}")
    
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch+1}] completed, Average Loss: {avg_loss:.4f}")

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dataloader, device, epoch)

# Evaluation
model.eval()
with torch.no_grad():
    for images, targets in test_dataloader:
        images = list(image.to(device) for image in images)
        outputs = model(images)

        for image, target, output in zip(images, targets, outputs):
            print(f"Image: {image.shape}, Target: {target}, Output: {output}")

# Save the model
torch.save(model.state_dict(), 'fasterrcnn_model.pth')

# Load the model for inference
model.load_state_dict(torch.load('fasterrcnn_model.pth'))
model.eval()

# Pick a random image from the test dataset
idx = np.random.randint(0, len(test_dataset))
image, target = test_dataset[idx]

# Move the image to the device
image = image.to(device)

model.eval()
with torch.no_grad():
    prediction = model([image])

print(f"Selected image index: {idx}")
print(f"Prediction: {prediction}")
print(f"Ground Truth: {target}")
