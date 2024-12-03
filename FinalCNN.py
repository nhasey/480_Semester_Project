import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

# Setting up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'sammynouadir'
os.environ['KAGGLE_KEY'] = 'f62bf71a470603e17116d3aa1843768b'

import kagglehub
#finding the path so we can get to test and train real and fake no longer needed 
#path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
#print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5647, 0.4770, 0.4273), (0.2724, 0.2619, 0.2676))
])

#Getting the path to the exact file needed as there is additional files we dont need and i dont want to deal with them you may need to replace it with yours
dataset_path = "/Users/oussamanouadir/.cache/kagglehub/datasets/xhlulu/140k-real-and-fake-faces/versions/2/real_vs_fake/real-vs-fake"

# Load the training dataset
train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the testing dataset
test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




# Dataset loaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class VerifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64* 8 * 8, 2)  
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten dynamically
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        #getting softmax
        return x
    
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Initialize the model, criterion, and optimizer
cnn_model = VerifierCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

num_epochs = 1

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train(cnn_model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
# Final evaluation
test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
print("\nFinal Evaluation on Test Set:")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

torch.save(cnn_model.state_dict(), 'cnn_weights.pth')

