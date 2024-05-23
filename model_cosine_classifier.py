import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        
        # Remove the original fully connected layer (classifier) of ResNet
        self.resnet.fc = nn.Identity()

        # Cosine Classifier
        self.cosine_classifier = nn.Linear(num_ftrs, num_classes, bias=False)
        nn.init.normal_(self.cosine_classifier.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # Feature extraction
        x = self.resnet(x)

        # Cosine classifier
        norm_x = F.normalize(x, p=2, dim=1)
        scores = self.cosine_classifier(norm_x)

        return scores