# Fruit2.py -- Uses a ResNet implementation of FruitCNN.
# Added random rotations and horizontal flips to images
# in transformations for better generalization.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import cv2 as cv
from PIL import UnidentifiedImageError

# Edge detection function
def edge_detection(image):
    gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 10, 435)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)

# Define transformations
transform = transforms.Compose([
    transforms.Lambda(edge_detection),
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class_name_id = {"Apple": 0, "Banana": 1, "Grape": 2, "Mango": 3, "Strawberry": 4}

class FruitDataset(Dataset):
    def __init__(self, root, transform=None, subset="train"):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        data_dir = os.path.join(root, subset)

        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if not os.path.isdir(label_path):
                continue

            for image_filename in os.listdir(label_path):
                image_path = os.path.join(label_path, image_filename)
                try:
                    Image.open(image_path).verify()
                    self.images.append(image_path)
                    self.labels.append(label)
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping invalid file: {image_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = class_name_id[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label
# Load Pretrained ResNet-50
class FruitResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(FruitResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Initialize the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FruitResNet50(num_classes=len(class_name_id)).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# Transformations (adjust input size for ResNet-50)
transform = transforms.Compose([
    transforms.Lambda(edge_detection),
    transforms.Resize((224, 224)),  # ResNet requires 224x224 inputs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
batch_size = 32
train_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="train")
val_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="valid")
test_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Beginning Training...")
# Training Loop
num_epochs = 4
best_val_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    # print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss / len(valid_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")


    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'resnet50_fruit_classifier.pth')

print("Finished Training!")

# Test the Model
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_accuracy = 100 * correct_test / total_test
print(f"Testing Accuracy: {test_accuracy:.2f}%")
