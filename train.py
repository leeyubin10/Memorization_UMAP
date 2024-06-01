import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model import CustomResNet
import os

# Define parameters
batch_size = 64
num_epochs = 100
learning_rate = 0.0001
num_classes = 10

def train_model():
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 dataset
    train_dataset = ImageFolder(root='cifar-10/train', transform=transform)
    val_dataset = ImageFolder(root='cifar-10/val', transform=transform)
    
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
    
    # Initialize lists to store loss and accuracy
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    weight_mat_epoch = {
    "epoch_0": None,
    "epoch_best": None,
    "epoch_last": None
    }

    # Train the model
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images, labels = images.to(device), (labels-1).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print("NaN loss detected")
                print(f"Outputs: {outputs}")
                print(f"Labels: {labels}")
                return

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
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

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
        val_accuracies.append(val_accuracy)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # Save the model checkpoint
        if epoch == 0:  # Save the checkpoint for the first epoch
            weight_mat_epoch["epoch_0"] = model.prototype_vectors.clone().cpu()
            torch.save(model.state_dict(), 'resnet18_epoch_0.pth')

        # Save the model checkpoint if it has the best test accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            weight_mat_epoch["epoch_best"] = model.prototype_vectors.clone().cpu()
            torch.save(model.state_dict(), 'resnet18_best.pth')

    # Save the final model
    weight_mat_epoch["epoch_last"] = model.prototype_vectors.clone().cpu()
    torch.save(model.state_dict(), 'resnet18_last.pth')
    
    torch.save(weight_mat_epoch, 'weight_mat_epoch.pth')
    
    # Plot and save graphs
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    
    # Plot training accuracy
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    
    # Plot validation accuracy
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_metrics_combined.png')
    plt.show()

if __name__ == "__main__":
    train_model()