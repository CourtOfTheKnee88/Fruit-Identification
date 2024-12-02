import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import numpy as np
import cv2 as cv

def edge_detection(image):
    # Convert to grayscale
    gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv.Canny(blurred, 10, 435)
    # Convert back to PIL Image with 3 channels
    edges = np.stack([edges] * 3, axis=-1)  # Duplicate grayscale into 3 channels
    return Image.fromarray(edges)

# Define transformations for the input data
transform = transforms.Compose([
    transforms.Lambda(edge_detection),
    transforms.Resize((200,200)),                  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class_name_id = {"Apple":0, "Banana":1, "Grape":2, "Mango":3, "Strawberry":4}

class FruitDataset(Dataset):
    def __init__(self, root, transform=None, subset="train"):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        # Load all images and labels
        if subset == "train":
            data_dir = os.path.join(root, "train")
        elif subset == "test":
            data_dir = os.path.join(root, "test")
        else:
            data_dir = os.path.join(root, "valid")
        for label in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, label)):
                continue
            else:
                for image_filename in os.listdir(os.path.join(data_dir, label)):
                    self.images.append(os.path.join(data_dir, label, image_filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = class_name_id[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# CNN model
class FruitCNN(nn.Module):
    def __init__(self):
        super(FruitCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 32 * 64 * 64)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
       
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model, Loss, and Optimizer
model = FruitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

batch_size = 32

# Create instances of the CustomDataset for training, validation, and testing
train_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="train")
val_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="val")
test_dataset = FruitDataset(root="/Users/nathanielserrano/Documents/GitHub/Fruit-Identification", transform=transform, subset="test")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# use the dataloader
for batch_id, (data, label) in enumerate(train_loader):
    print("Batch ID:" + str(batch_id))
    print("Data Shape:")
    print(data.shape)
    print("Label Shape:")
    print(len(label))
    break


