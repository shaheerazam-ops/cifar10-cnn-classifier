import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TransferResNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Load pretrained ResNet18
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False
        # unfreeze layer 4
        for param in self.model.parameters():
            param.requires_grad= True    

        # Replace ImageNet classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            10
        )

    def forward(self, x):

        return self.model(x)