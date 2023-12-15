import os
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from utils.DivisiveNormalization import OpponentChannelInhibition, DivisiveNormalizationCirincione
from utils.DoG import DoGConv2DLayer
from utils.LocallyConnected2d import LocallyConnected2d
from utils.PolarTransform import PolarTransform

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


class ConvDog(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, dilation: int = 1, bias=False):
        super(ConvDog, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.dog = DoGConv2DLayer(dog_channels=in_planes//4, k=2, stride=1, padding=2, bias=False)
    def forward(self, x):
        return self.dog(self.conv(x))

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3_dog(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding followed by DoG convolution"""
    return ConvDog(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3_local(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> LocallyConnected2d:
    """Local 3x3 convolution with padding"""
    return LocallyConnected2d(in_planes, out_planes, output_size=16, kernel_size=3, stride=stride, padding=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BrainBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    use_dog: bool = True
    divnorm = 'tuned'

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.use_dog:
            self.conv2 = conv3x3_dog(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.divnorm == 'tuned':
            self.dn = OpponentChannelInhibition(planes * self.expansion)
        elif self.divnorm == 'cirincione':
            self.dn = DivisiveNormalizationCirincione(planes * self.expansion, 3)
        else:
            self.dn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return self.dn(out)


class BrainBottleneckLocal(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    use_local: bool = True
    divnorm = 'tuned'

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.use_local:
            self.conv2 = conv3x3_local(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if self.divnorm == 'tuned':
            self.dn = OpponentChannelInhibition(planes * self.expansion)
        elif self.divnorm == 'cirincione':
            self.dn = DivisiveNormalizationCirincione(planes * self.expansion, 3)
        else:
            self.dn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return self.dn(out)


class ResNetComposite(nn.Module):

    use_polar_transform: bool = True

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_micro_layer([BrainBottleneck,BrainBottleneck,BrainBottleneckLocal], 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        img_size = 32
        out_size = 16
        input_h, input_w = img_size, img_size
        output_h, output_w = out_size, out_size
        foveal_radius = 12
        max_dist = np.sqrt((input_h/2)**2 + (input_w/2)**2)
        foveal_dists = np.arange(0, foveal_radius)
        periphery_dists = [12, 13, 14, 16, 18, 20, 23]
        radius_bins = list(foveal_dists) + list(periphery_dists)
        angle_bins = list(np.linspace(0, 2*np.pi, img_size+1-2))
        self.lpp = PolarTransform(input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation='linear', subbatch_size=32)
        self.maxpool = None
        if not self.use_polar_transform:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_micro_layer(
        self,
        blockset,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * blockset[0].expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * blockset[0].expansion, stride),
                norm_layer(planes * blockset[0].expansion),
            )

        layers = []
        layers.append(
            blockset[0](
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * blockset[0].expansion
        for i in range(1, blocks):
            layers.append(
                blockset[i](
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        clf_x = self.conv1(x)
        clf_x = self.bn1(clf_x)
        clf_x = self.relu(clf_x)
        if self.use_polar_transform:
            clf_x = self.lpp(clf_x)
        else:
            clf_x = self.maxpool(clf_x)
        clf_x = self.layer1(clf_x)
        clf_x = self.layer2(clf_x)
        clf_x = self.layer3(clf_x)
        clf_x = self.layer4(clf_x)
        clf_x = self.avgpool(clf_x)
        clf_x = torch.flatten(clf_x, 1)
        clf_x = self.fc(clf_x)
        return clf_x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_composite(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights,
) -> ResNetComposite:
    model = ResNetComposite(block, layers)
    if weights is not None:
        model.load_state_dict(weights)
    return model

def resnet50_composite_a(*, weights = None) -> ResNetComposite:
    """ResNet-50 v1.5 supplemented with center-surround antagonism, local RFs, 
       tuned_normalization, and cortical magnification
    """
    BrainBottleneck.use_dog = True
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_b(*, weights = None) -> ResNetComposite:
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_c(*, weights = None) -> ResNetComposite:
    BrainBottleneck.use_dog = True
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = None
    BrainBottleneckLocal.divnorm = None
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_d(*, weights = None) -> ResNetComposite:
    """ResNet-50 v1.5 supplemented with center-surround antagonism, 
       tuned_normalization, and cortical magnification
    """
    BrainBottleneck.use_dog = True
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    ResNetComposite.use_polar_transform = False
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_e(*, weights = None) -> ResNetComposite:
    """ResNet-50 v1.5 supplemented with tuned_normalization and cortical magnification
    """
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = False
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    ResNetComposite.use_polar_transform = True
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_f(*, weights = None) -> ResNetComposite:
    """ResNet-50 v1.5 supplemented with local RF and cortical magnification
    """
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = None
    BrainBottleneckLocal.divnorm = None
    ResNetComposite.use_polar_transform = True
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_g(*, weights = None) -> ResNetComposite:
    """
    ResNet-50 v1.5 supplemented with local RF and tuned normalization
    """
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = True
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    ResNetComposite.use_polar_transform = False
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_composite_model(model_variant='a'):
    assert model_variant in ['a','b','c','d','e','f','g'], 'Invalid composite model selection'
    if model_variant == 'a':
        model = resnet50_composite_a()
    elif model_variant == 'b':
        model = resnet50_composite_b()
    elif model_variant == 'c':
        model = resnet50_composite_c()
    elif model_variant == 'd':
        model = resnet50_composite_d()
    elif model_variant == 'e':
        model = resnet50_composite_e()
    elif model_variant == 'f':
        model = resnet50_composite_f()
    elif model_variant == 'g':
        model = resnet50_composite_g()
    return model

def resnet50_tuned_norm(*, weights = None) -> ResNetComposite:
    """
    ResNet-50 v1.5 supplemented with tuned_normalization
    """
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = False
    BrainBottleneck.divnorm = 'tuned'
    BrainBottleneckLocal.divnorm = 'tuned'
    ResNetComposite.use_polar_transform = False
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)

def resnet50_div_norm(*, weights = None) -> ResNetComposite:
    """
    ResNet-50 v1.5 supplemented with cirincione divisive normalization
    """
    BrainBottleneck.use_dog = False
    BrainBottleneckLocal.use_local = False
    BrainBottleneck.divnorm = 'cirincione'
    BrainBottleneckLocal.divnorm = 'cirincione'
    ResNetComposite.use_polar_transform = False
    return _resnet_composite(Bottleneck, [3, 4, 6, 3], weights)
    