import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split

# Setting up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'sammynouadir'
os.environ['KAGGLE_KEY'] = 'f62bf71a470603e17116d3aa1843768b'

# Loading datasets from Kaggle using kagglehub
import kagglehub
pathAI = kagglehub.dataset_download("chelove4draste/10k-ai-generated-faces")
pathHuman = kagglehub.dataset_download("ashwingupta3012/human-faces")

# Data handling and transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5647, 0.4770, 0.4273), (0.2724, 0.2619, 0.2676))
])

# Combining datasets using ConcatDataset
combined_dataset = ConcatDataset([
    torchvision.datasets.ImageFolder(root=pathAI, transform=transform),
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

def evaluate(model,loader,criterion):
    model.eval()
    running_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in loader:
            outputs=model(images)
            loss=criterion(outputs,labels)

            running_loss+=loss.item()*images.size(0)
            _, predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        epoch_loss=running_loss/total
        epoch_acc=100*correct/total
        return epoch_loss,epoch_acc

cnn_model=VerifierCNN()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(cnn_model.parameters(),lr=0.001)

num_epochs=1

for epoch in range(num_epochs):
    train__loss,train__acc=train(cnn_model,train_loader,criterion,optimizer)
    test__loss,test__acc=evaluate(cnn_model,test_loader,criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}]-'
        f'Train Loss: {train__loss:.4f}, Train Acc: {train__acc:.2f}%-'
        f'Test Loss: {test__loss:.4f}, Test Acc: {test__acc:.2f}%')
    
    test__loss, test__acc = evaluate(cnn_model, test_loader,criterion)
    print("\nFinal Evaluation on Test Set:")
    print(f"Test Loss: {test__loss:.4f}, Test Accuracy: {test__acc:.2f}%")





