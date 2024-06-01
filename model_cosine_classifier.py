import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pdb

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        
        # Remove the original fully connected layer (classifier) of ResNet
        self.resnet.fc = nn.Identity()

        # Cosine Classifier
        self.prototype_vectors = nn.Parameter(torch.randn(num_classes, 512))  # 512는 ResNet18의 출력 특징 차원

    def forward(self, x):
        # Feature extraction
        x = self.resnet(x)
        #pdb.set_trace()
        # Cosine classifier
        features = x / x.norm(dim=1, keepdim=True)  # Normalize the feature vectors
        prototypes = self.prototype_vectors / self.prototype_vectors.norm(dim=1, keepdim=True)  # Normalize the prototype vectors
        cosine_sim = torch.matmul(features, prototypes.t())  # Cosine similarity between features and prototypes
        return cosine_sim