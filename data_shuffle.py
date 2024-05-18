import numpy as np
import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from distribution import visualize_dataset

# train 폴더 생성
os.makedirs('train', exist_ok=True)

# 클래스 이름 리스트
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define parameters
num_classes = 10
random_label_percentage = 0.5  # 50% of data to randomly relabel

def process_dataset():
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Define the file name to store label changes
    label_changes_file = 'label_changes.txt'

    # Load CIFAR-10 dataset
    train_dataset = ImageFolder(root='cifar-10/train', transform=transform)

    # Get indices for each class
    class_indices = {label: np.where(np.array(train_dataset.targets) == label)[0] 
                     for label in range(num_classes)}

    # Initialize list to count random labels for each class
    random_label_counts = [0] * num_classes

    original_indices = []
    random_label_indices = []
    original_labels = {}
    random_labels = {}

    for label, indices in class_indices.items():
        # Randomly select 50% indices from each class
        random_indices = np.random.choice(indices, size=int(len(indices) * random_label_percentage), replace=False)

        original_indices.extend([idx for idx in indices if idx not in random_indices])
        random_label_indices.extend(random_indices)

        # Change labels for selected indices
        for idx in random_indices:
            new_label = np.random.randint(num_classes)
            while new_label == label:  # Change to a different label from the original one
                new_label = np.random.randint(num_classes)
            train_dataset.targets[idx] = new_label
            random_label_counts[new_label] += 1

            # Save original and new labels
            original_labels[idx] = label
            random_labels[idx] = new_label

            # Save label changes to a file
            with open(label_changes_file, 'a') as f:
                f.write(f"Original Label: {label}, New Label: {new_label}\n")
                
    visualize_dataset(train_dataset, num_classes, random_label_counts)

    # Save original labeled images
    for class_idx in range(num_classes):
        os.makedirs(os.path.join('train_random/original_label', class_names[class_idx]), exist_ok=True)

    for idx in original_indices:
        path, label = train_dataset.samples[idx]
        class_folder = class_names[label]
        shutil.copy(path, os.path.join('train_random/original_label', class_folder, os.path.basename(path)))

    # Save random labeled images
    for class_idx in range(num_classes):
        os.makedirs(os.path.join('train_random/random_label', class_names[class_idx]), exist_ok=True)

    for idx in random_label_indices:
        path, _ = train_dataset.samples[idx]
        new_label = random_labels[idx]
        class_folder = class_names[new_label]
        shutil.copy(path, os.path.join('train_random/random_label', class_folder, os.path.basename(path)))

if __name__ == "__main__":
    process_dataset()
