import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50
from utils.LocallyConnected2d import LocallyConnected2d

def resnet50_local_rf():
    model = resnet50()
    model.layer1[2].conv2 = torch.nn.Sequential(
        torch.nn.Dropout2d(p=0.5, inplace=False),
        LocallyConnected2d(64, 64, 16, kernel_size=3, stride=1, padding=1)
    )
    return model