import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import CustomResNet

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


# Load CIFAR-10 dataset
root_dir = 'cifar-10/train'
class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != '.ipynb_checkpoints']
test_dataset = ImageFolder(root=root_dir, transform=transform)
test_dataset.classes = class_names
test_dataset.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

# Define the number of data points to visualize per class
num_points_per_class = 1000  # You can adjust this number as needed

# Create data loader
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(test_dataset.classes)

checkpoint_file = 'resnet18_last.pth'
        
# Load the model checkpoint
model = CustomResNet().to(device)
model.load_state_dict(torch.load(checkpoint_file, map_location=device))
model.eval()
softmax = torch.nn.Softmax(dim=1)

for img, label in test_loader:
    img = img.to(device)
    output = model(img)
    prob = softmax(output)
    
    # Debugging: Print shapes
    print("Number of classes:", len(test_dataset.classes))
    print("Shape of prob tensor:", prob.shape)
    
    break
    
# 클래스별 확률 시각화
for i in range(len(img)):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(test_dataset.classes)-1), prob[i].detach().cpu().numpy())
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title(f'Class Probabilities for Image {i+1}')
    plt.xticks(np.arange(len(test_dataset.classes)), test_dataset.classes, rotation=90)
    plt.show()
    # 필요한 경우 파일로 저장
    plt.savefig(f'probabilities_image_{i+1}.png')