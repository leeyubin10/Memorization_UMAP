import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tsne_model import CustomResNet

# Define the device
if torch.cuda.is_available():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    device = torch.device("cuda:0")
    print('We will use the GPU:', torch.cuda.current_device())
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define UMAP
umap_emb = umap.UMAP(n_neighbors=3, min_dist=0.3, n_components=2, metric='euclidean')

# Load CIFAR-10 dataset
test_dataset = test_dataset = ImageFolder(root='cifar-10/train', transform=transform)

# Create data loader
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define epochs to visualize
epochs_to_visualize = [0, 'best', 'last']

# Loop through each epoch checkpoint
for epoch in epochs_to_visualize:
    if epoch == 'best':
        checkpoint_file = 'new_resnet18_best.pth'
    elif epoch == 'last':
        checkpoint_file = 'new_resnet18_last.pth'
    else:
        checkpoint_file = f"new_resnet18_epoch_{epoch}.pth"
        
    # Load the model checkpoint
    model = CustomResNet().to(device)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()

    # Extract features and labels
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate features and labels from all epochs
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Apply UMAP
    umap_results = umap_emb.fit_transform(all_features)
    
    for i, epoch_file in enumerate(epochs_to_visualize):
        for j in range(10):
            idx = np.where(all_labels == j)
            plt.scatter(umap_results[idx, 0], umap_results[idx, 1], cmap='tab10', s=5)
        plt.colorbar()
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'UMAP Visualization of ResNet-18 Features (Epoch {epoch})')
        plt.legend()
        
        # Save the plot as an image file
        save_dir = 'umap_visualizations'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'umap_epoch_{epoch}.png'))
        plt.close()  # Close the plot to release memory

print("UMAP visualizations saved.")
