import torch
from sklearn.manifold import TSNE
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

# Define t-SNE
tsne = TSNE(n_components=2, random_state=0)

# Load CIFAR-10 dataset
test_dataset = test_dataset = ImageFolder(root='cifar-10/train', transform=transform)

# Define the number of data points to visualize per class
num_points_per_class = 1000  # You can adjust this number as needed

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
    
    # Reduce the number of data points per class
    reduced_features = []
    reduced_labels = []
    for class_label in range(10):  # Assuming 10 classes
        class_indices = np.where(all_labels == class_label)[0]
        selected_indices = np.random.choice(class_indices, size=num_points_per_class, replace=False)
        reduced_features.append(all_features[selected_indices])
        reduced_labels.append(all_labels[selected_indices])
    
    reduced_features = np.concatenate(reduced_features, axis=0)
    reduced_labels = np.concatenate(reduced_labels, axis=0)
    
    # Apply t-SNE
    tsne_results = tsne.fit_transform(reduced_features)
    
    plt.figure(figsize=(10, 8))
    for j in range(10):
        idx = np.where(reduced_labels == j)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], marker='.', label=str(j), alpha=0.5)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Visualization of ResNet-18 Features (Epoch {epoch})')
    plt.legend()
        
    # Save the plot as an image file
    save_dir = 'tsne_visualizations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()  # Close the plot to release memory

print("t-SNE visualizations saved.")
