import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Assuming you have a dataset of human faces
data_dir = r"C:\Users\natha\480Project\480_Semester_Project\Data\128Formatted"

# Create a basic transformation without normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Compute mean and std of the dataset
def get_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    iterator = 0
    for data, _ in loader:
        
        print("On batch: ", iterator)
        print("Batch Shape: ", data.shape)
        iterator = iterator+1
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

mean, std = get_mean_and_std(loader)
print(f"Mean: {mean}, Std: {std}")
