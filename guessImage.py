import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 2)  # Two output classes: real and fake
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten the output for the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load the entire model
model_path = 'cnn_complete_model.pth'
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define the transformation (same as used during training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5647, 0.4770, 0.4273), (0.2724, 0.2619, 0.2676))
])

# Function to preprocess an image and make a prediction
def predict_image(image_path, model):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Apply the transformation
    image = transform(image)
    # Add a batch dimension (1, C, H, W)
    image = image.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Map label to class name
    class_names = ['fake', 'real']  # Assuming 'fake' is 0 and 'real' is 1 based on folder structure
    predicted_class = class_names[predicted.item()]
    
    return predicted_class

# Test the function with a new image
image_path = r"C:\Users\natha\OneDrive\Desktop\Aidan Test\128\horse.jpeg"  # Replace with the actual path to your image
predicted_class = predict_image(image_path, model)
print(f'The predicted class is: {predicted_class}')
