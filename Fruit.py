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
    
# CNN model
class FruitCNN(nn.Module):
   def __init__(self, num_classes=5):
    super(FruitCNN, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.layer2 = nn.Sequential(
        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.layer3 = nn.Sequential(
        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU())
    self.layer4 = nn.Sequential(
        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU())
    self.layer5 = nn.Sequential(
        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2))
    self.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(9216, 4096),
        nn.ReLU())
    self.fc1 = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU())
    self.fc2= nn.Sequential(
        nn.Linear(4096, num_classes))

batch_size = 20

train_dataset = FruitDataset(root="C:/Users/court/Desktop/Fruit-Identification", transform=transform, subset="train")
val_dataset = FruitDataset(root="C:/Users/court/Desktop/Fruit-Identification", transform=transform, subset="valid")
test_dataset = FruitDataset(root="C:/Users/court/Desktop/Fruit-Identification", transform=transform, subset="test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class FruitCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FruitCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU()
        # )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU()
        # )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )

        self.flatten_size = self._get_conv_output_size((3, 200, 200))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def _get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        # output = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(dummy_input)))))
        output = self.layer2(self.layer1(dummy_input))
        return output.numel()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FruitCNN().to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer  = optim.SGD(model.parameters(), lr=0.1)

print("Beginning Training...")

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation Loop
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            val_loss += loss.item()

    val_accuracy = 100 * correct_val / total_val
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss / len(valid_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print('Finished Training')

# Save the Model
torch.save(model.state_dict(), 'fruit_cnn.pth')

# Testing Loop
model.eval()
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Testing Accuracy: {test_accuracy:.2f}%")
