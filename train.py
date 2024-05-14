import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model import CustomResNet

from distribution import visualize_dataset

# Define parameters
batch_size = 64
num_epochs = 100
learning_rate = 0.001
num_classes = 10
random_label_percentage = 0.5  # 50% of data to randomly relabel
val_split = 0.2  # 20% of train dataset for validation

def train_model():
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define the file name to store label changes
    label_changes_file = 'label_changes.txt'

    # Load CIFAR-10 dataset
    train_dataset = ImageFolder(root='cifar-10/train', transform=transform)
    val_dataset = ImageFolder(root='cifar-10/val', transform=transform)
    
    # Get indices for each class
    class_indices = {label: np.where(np.array(train_dataset.targets) == label)[0] 
                     for label in range(num_classes)}
    
    # Initialize list to count random labels for each class
    random_label_counts = [0] * num_classes
    
    for label, indices in class_indices.items():
        # Randomly select 50% indices from each class
        random_indices = np.random.choice(indices, size=int(len(indices) * random_label_percentage), replace=False)

        # Change labels for selected indices
        for idx in random_indices:
            new_label = np.random.randint(num_classes)
            while new_label == label:  # Change to a different label from the original one
                new_label = np.random.randint(num_classes)
            train_dataset.targets[idx] = new_label
            random_label_counts[new_label] += 1
            
            # Save label changes to a file
            with open(label_changes_file, 'a') as f:
                f.write(f"Original Label: {label}, New Label: {new_label}\n")
            
    visualize_dataset(train_dataset, num_classes, random_label_counts)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the device
    if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        device = torch.device("cuda:0")
        print('We will use the GPU:', torch.cuda.current_device())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Define the model
    model = CustomResNet().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Compute statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Compute epoch statistics
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_predictions / total_predictions

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Test the model on the test dataset
        model.eval()
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_predictions
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Save the model checkpoint
        if epoch == 0:  # Save the checkpoint for the first epoch
            torch.save(model.state_dict(), 'resnet18_epoch_0.pth')

        # Save the model checkpoint if it has the best test accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'resnet18_best.pth')

    # Save the final model
    torch.save(model.state_dict(), 'resnet18_last.pth')

if __name__ == "__main__":
    train_model()