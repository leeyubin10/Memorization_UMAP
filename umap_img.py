import torch
import umap
import matplotlib.pyplot as plt
import os

from model import CustomResNet
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np

# Define UMAP
umap_emb = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, metric='euclidean')

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
test_dataset = ImageFolder(root='cifar-10/test', transform=transform)

# Create data loader
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define epochs to visualize
epochs_to_visualize = [0, 'best', 'last']  # Add 'best' and 'last' if needed

for epoch in epochs_to_visualize:
    if epoch == 'best':
        checkpoint_file = 'resnet18_best.pth'
    elif epoch == 'last':
        checkpoint_file = 'resnet18_last.pth'
    else:
        checkpoint_file = f"resnet18_epoch_{epoch}.pth"

    model = CustomResNet()
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()

    # Extract features
    all_features = []
    all_labels = []
    for images, labels in test_loader:
        features = model(images)
        all_features.append(features.detach().numpy())
        all_labels.append(labels.numpy())

    # Flatten features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Apply UMAP
    umap_results = umap_emb.fit_transform(all_features)

    # Plot UMAP results with bright colors
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar()
    plt.title(f'UMAP Visualization of ResNet-18 Features (Epoch {epoch})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # Save the plot as an image file
    save_dir = 'umap_visualizations_bright'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'umap_epoch_{epoch}.png'))
    plt.close()  # Close the plot to release memory
    
print("UMAP visualizations with seaborn scatterplot saved.")