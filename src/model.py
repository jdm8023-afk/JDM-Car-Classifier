import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    """
    Load pretrained ResNet50 and replace final layer.
    """

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
