import torch
import torch.nn as nn
from torchvision.models import resnet18

# Define custom ResNet model with feature extraction hook
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.features = None

        # Register hook to extract features
        self.resnet.layer4.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.resnet.fc(x)

        return x