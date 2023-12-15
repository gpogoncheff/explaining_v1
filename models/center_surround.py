import torch
import torch.nn as nn
from torchvision.models import resnet50
from utils.DoG import DoGConv2DLayer

def resnet50_center_surround():
    model = resnet50()
    model.layer1[0].conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        DoGConv2DLayer(dog_channels=16, k=2, stride=1, padding=2, bias=False)
    )
    model.layer1[1].conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        DoGConv2DLayer(dog_channels=16, k=2, stride=1, padding=2, bias=False)
    )
    model.layer1[2].conv2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        DoGConv2DLayer(dog_channels=16, k=2, stride=1, padding=2, bias=False)
    )
    return model