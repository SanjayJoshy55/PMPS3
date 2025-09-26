# model_loader.py

import torch
import torch.nn as nn
from torchvision import models
from timm import create_model

def load_spiral_model():
    """Defines and loads the trained ResNet18 model for spirals."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))
    model.load_state_dict(torch.load("best_spiral_resnet18.pth", map_location="cpu"))
    model.eval()
    return model

class ConvNeXtBase(nn.Module):
    def __init__(self):
        super(ConvNeXtBase, self).__init__()
        self.model = create_model("convnext_base", pretrained=False, num_classes=2)
    def forward(self, x):
        return self.model(x)

def load_wave_model():
    """Defines and loads the trained ConvNeXt-Base model for waves."""
    model = ConvNeXtBase()
    model.load_state_dict(torch.load("best_convnext_base_model.pth", map_location="cpu"))
    model.eval()
    return model