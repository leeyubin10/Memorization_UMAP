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
original_dataset = ImageFolder(root='original_label', transform=transform)
random_dataset = ImageFolder(root='random_label', transform=transform)

# Define the number of data points to visualize per class
num_points_per_class = 1000  # You can adjust this number as needed

# Create data loader
batch_size = 64
original_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False)
random_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=False)

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
    def extract_features_and_labels(loader):
        features = []
        labels = []
        with torch.no_grad():
            for images, labels_batch in loader:
                images = images.to(device)
                features_batch = model(images)
                features.append(features_batch.cpu().numpy())
                labels.append(labels_batch.numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    original_features, original_labels = extract_features_and_labels(original_loader)
    random_features, random_labels = extract_features_and_labels(random_loader)
    
    # Reduce the number of data points per class
    def reduce_data(features, labels, num_points_per_class):
        reduced_features = []
        reduced_labels = []
        for class_label in range(10):  # Assuming 10 classes
            class_indices = np.where(labels == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=num_points_per_class, replace=False)
            reduced_features.append(features[selected_indices])
            reduced_labels.append(labels[selected_indices])
        reduced_features = np.concatenate(reduced_features, axis=0)
        reduced_labels = np.concatenate(reduced_labels, axis=0)
        return reduced_features, reduced_labels

    original_features, original_labels = reduce_data(original_features, original_labels, num_points_per_class)
    random_features, random_labels = reduce_data(random_features, random_labels, num_points_per_class)
    
    # Apply t-SNE
    all_features = np.concatenate((original_features, random_features), axis=0)
    tsne_results = tsne.fit_transform(all_features)
    
    original_tsne_results = tsne_results[:len(original_features)]
    random_tsne_results = tsne_results[len(original_features):]
    
    plt.figure(figsize=(15, 13))
    colors = plt.cm.get_cmap('tab10', 10)  # Get a colormap with 10 distinct colors
    for j in range(10):
        original_idx = np.where(original_labels == j)
        random_idx = np.where(random_labels == j)
        
        plt.scatter(original_tsne_results[original_idx, 0], original_tsne_results[original_idx, 1], 
                    marker='.', color=colors(j), label=f'Class {j}', alpha=0.5)
        plt.scatter(random_tsne_results[random_idx, 0], random_tsne_results[random_idx, 1], 
                    marker='x', color=colors(j), label=f'Class {j} (Random)', alpha=0.5)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Visualization of ResNet-18 Features (Epoch {epoch})')
    
    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), fontsize='small')
        
    # Save the plot as an image file
    save_dir = 'tsne_visualizations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()  # Close the plot to release memory

print("t-SNE visualizations saved.")
