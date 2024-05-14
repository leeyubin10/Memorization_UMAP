import torch
import torch.nn as nn
from torchvision.models import resnet18

# Define custom ResNet model with feature extraction hook
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.features = None
        
        # Replace the FC layer with Identity
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        # Forward pass through ResNet layers
        x = self.resnet(x)
        return x