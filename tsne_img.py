import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns

from model import CustomResNet
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np

# Define t-SNE
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)

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

# Set seaborn style
sns.set_style("whitegrid")

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

    # Apply t-SNE
    tsne_results = tsne.fit_transform(all_features)

    # Plot t-SNE results with seaborn scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=all_labels, palette="bright", legend='full', s=50)
    plt.title(f't-SNE Visualization of ResNet-18 Features (Epoch {epoch})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    # Save the plot as an image file
    save_dir = 'tsne_visualizations_seaborn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()  # Close the plot to release memory

print("t-SNE visualizations with seaborn scatterplot saved.")