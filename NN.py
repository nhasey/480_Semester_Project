import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split
import kagglehub
import torch.nn.functional as F
from collections import Counter

# Setting up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'sammynouadir'
os.environ['KAGGLE_KEY'] = 'f62bf71a470603e17116d3aa1843768b'

# Loading datasets from Kaggle using kagglehub

#pathAI = kagglehub.dataset_download("chelove4draste/10k-ai-generated-faces")
pathHuman = kagglehub.dataset_download("kaustubhdhote/human-faces-dataset")
if os.name == 'nt':  #windows
    home_dir = os.environ.get('USERPROFILE', '')  
else:  #mac
    home_dir = os.environ.get('HOME', '') 

pathHuman = os.path.join(home_dir, '.cache', 'kagglehub', 'datasets', 'kaustubhdhote', 'human-faces-dataset', 'versions', '1', 'Human Faces Dataset')

# Data handling and transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5424, 0.4509, 0.3992), (0.2629, 0.2425, 0.2437))
])

# Combining datasets using ConcatDataset
combined_dataset = ConcatDataset([
    #torchvision.datasets.ImageFolder(root=pathAI, transform=transform),
    torchvision.datasets.ImageFolder(root=pathHuman, transform=transform)
])

# Splitting the combined dataset into 80% train and 20% test
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
#filling train with 80% and test with 20% from combined dataset
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Dataset loaders
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class_counts = Counter()
for _, labels in train_loader:
    class_counts.update(labels.numpy())

print("Class distribution in training set:", class_counts)

class  VerifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,8,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(8,16,kernel_size=3,padding=1)
        self.fc1=nn.Linear(16*32*32,2)
    
    def forward(self, x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=self.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(-1,16*32*32)
        x=self.fc1(x)
        return x
    
def train(model, loader, criterion,optimizer):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for images, labels in loader:
        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    epoch_loss=running_loss/total
    epoch_acc=100*correct/total
    return epoch_loss,epoch_acc



def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_confidences = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            confidences = probabilities * 100  # Convert to percentages
            all_confidences.append(confidences)
            
            # Make sure the following print statements are consistently indented with the rest of the function
            print("Raw logits (first 5 samples):", outputs[:5])
            print("Softmax probabilities (first 5 samples):", probabilities[:5])

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    all_confidences = torch.cat(all_confidences, dim=0)
    
    return epoch_loss, epoch_acc, all_confidences


cnn_model=VerifierCNN()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(cnn_model.parameters(),lr=0.0001)

num_epochs=1

for epoch in range(num_epochs):
    train__loss, train__acc = train(cnn_model, train_loader, criterion, optimizer)
    test__loss, test__acc, test_confidences = evaluate(cnn_model, test_loader, criterion)

    print(f'Epoch [{epoch+1}/{num_epochs}]-'
          f'Train Loss: {train__loss:.4f}, Train Acc: {train__acc:.2f}%-'
          f'Test Loss: {test__loss:.4f}, Test Acc: {test__acc:.2f}%')
    
    # Display confidences for a few test predictions
    print("\nClass-wise confidence percentages for some test predictions:")
    for i, confidence in enumerate(test_confidences[:5]):  # Show first 5 test samples
        print(f"Sample {i+1}: Class 0: {confidence[0]:.2f}%, Class 1: {confidence[1]:.2f}%")
    
# Final evaluation
test__loss, test__acc, test_confidences = evaluate(cnn_model, test_loader, criterion)
print("\nFinal Evaluation on Test Set:")
print(f"Test Loss: {test__loss:.4f}, Test Accuracy: {test__acc:.2f}%")
