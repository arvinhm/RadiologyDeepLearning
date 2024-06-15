import torch
import torch.nn as nn
import torchvision.models as models
import timm
from collections import OrderedDict

# Define 3D versions of basic building blocks
class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
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

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
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

class ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=2, num_classes=1):
        super(ResNet3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18_3d(in_channels, num_classes):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels, num_classes)

def resnet34_3d(in_channels, num_classes):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, num_classes)

def resnet50_3d(in_channels, num_classes):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, num_classes)

def resnet101_3d(in_channels, num_classes):
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels, num_classes)

def resnet152_3d(in_channels, num_classes):
    return ResNet3D(Bottleneck3D, [3, 8, 36, 3], in_channels, num_classes)

# Define DenseNet3D
class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer3D(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x

class _DenseLayer3D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3D, self).forward(x)
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, in_channels=2, num_classes=1):
        
        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_layers=num_layers, num_input_features=num_features,
                               bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition3D(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121_3d(in_channels, num_classes):
    return DenseNet3D(32, (6, 12, 24, 16), 64, 4, 0, in_channels, num_classes)

def densenet169_3d(in_channels, num_classes):
    return DenseNet3D(32, (6, 12, 32, 32), 64, 4, 0, in_channels, num_classes)

def densenet201_3d(in_channels, num_classes):
    return DenseNet3D(32, (6, 12, 48, 32), 64, 4, 0, in_channels, num_classes)

def densenet264_3d(in_channels, num_classes):
    return DenseNet3D(32, (6, 12, 64, 48), 64, 4, 0, in_channels, num_classes)

# Inception3D
class InceptionResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=10):
        super(InceptionResNet3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def inception_resnet3d(in_channels, num_classes):
    return InceptionResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, num_classes)

# MobileNet3D
class MobileNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MobileNet3D, self).__init__()
        self.in_channels = in_channels
        self.model = timm.create_model('mobilenetv2_100', pretrained=True)
        self.model.conv_stem = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.bn1 = nn.BatchNorm3d(32)
        self._replace_2d_with_3d()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._get_classifier_input_features(), num_classes),
        )

    def _replace_2d_with_3d(self):
        replace_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=(module.kernel_size[0], module.kernel_size[0], module.kernel_size[0]),
                    stride=(module.stride[0], module.stride[0], module.stride[0]),
                    padding=(module.padding[0], module.padding[0], module.padding[0]),
                    bias=module.bias is not None
                )
                replace_dict[name] = new_module
            elif isinstance(module, nn.BatchNorm2d):
                new_module = nn.BatchNorm3d(module.num_features)
                replace_dict[name] = new_module

        # Apply the replacements
        for name, new_module in replace_dict.items():
            self._set_module_by_name(name, new_module)

    def _set_module_by_name(self, name, new_module):
        components = name.split('.')
        module = self.model
        for comp in components[:-1]:
            module = getattr(module, comp)
        setattr(module, components[-1], new_module)

    def _get_classifier_input_features(self):
        # Dummy forward pass to get the size of the features
        dummy_input = torch.zeros(1, self.in_channels, 16, 112, 112)  # Adjust the dimensions as needed
        features = self.model.forward_features(dummy_input)
        return features.shape[1]

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.mean([2, 3, 4])  # Global average pooling over spatial dimensions
        x = self.model.classifier(x)
        return x

def mobilenet3d(in_channels, num_classes):
    return MobileNet3D(in_channels, num_classes)


# EfficientNet3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Conv3dSame(nn.Conv3d):
    def forward(self, x):
        return F.conv3d(
            x, self.weight, self.bias, self.stride,
            self._padding(x), self.dilation, self.groups)

    def _padding(self, x):
        input_d, input_h, input_w = x.size()[-3:]
        stride_d, stride_h, stride_w = self.stride
        kernel_d, kernel_h, kernel_w = self.kernel_size
        dilation_d, dilation_h, dilation_w = self.dilation

        out_d = (input_d + stride_d - 1) // stride_d
        out_h = (input_h + stride_h - 1) // stride_h
        out_w = (input_w + stride_w - 1) // stride_w

        pad_d = max((out_d - 1) * stride_d + (kernel_d - 1) * dilation_d + 1 - input_d, 0)
        pad_h = max((out_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
        pad_w = max((out_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)

        # Ensure padding is always even
        if pad_d % 2 != 0:
            pad_d += 1
        if pad_h % 2 != 0:
            pad_h += 1
        if pad_w % 2 != 0:
            pad_w += 1
        
        return pad_d // 2, pad_h // 2, pad_w // 2


class MBConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MBConv3d, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm3d(mid_channels)
        self.depthwise_conv = Conv3dSame(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.project_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.swish = nn.SiLU()
        
        self.skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        residual = x

        x = self.swish(self.bn0(self.expand_conv(x)))
        x = self.swish(self.bn1(self.depthwise_conv(x)))
        x = self.bn2(self.project_conv(x))

        if self.skip:
            x = x + residual

        return x

class EfficientNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(EfficientNet3D, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', Conv3dSame(in_channels, 32, kernel_size=3, stride=2)),
            ('bn1', nn.BatchNorm3d(32)),
            ('swish', nn.SiLU()),
            ('mbconv1', MBConv3d(32, 16, 1, 1)),
            ('mbconv2', MBConv3d(16, 24, 2, 6)),
            ('mbconv3', MBConv3d(24, 40, 2, 6)),
            ('mbconv4', MBConv3d(40, 80, 2, 6)),
            ('mbconv5', MBConv3d(80, 112, 1, 6)),
            ('mbconv6', MBConv3d(112, 192, 2, 6)),
            ('mbconv7', MBConv3d(192, 320, 1, 6)),
            ('conv2', Conv3dSame(320, 1280, kernel_size=1)),
            ('bn2', nn.BatchNorm3d(1280)),
            ('swish2', nn.SiLU())
        ]))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def efficientnet3d_b0(in_channels, num_classes):
    model = EfficientNet3D(in_channels=in_channels, num_classes=num_classes)
    return model

def efficientnet3d_b7(in_channels, num_classes):
    model = EfficientNet3D(in_channels=in_channels, num_classes=num_classes)
    return model



# VGG3D
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv3DSame, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        pad_d = max((self.kernel_size[0] - 1) // 2, 0)
        pad_h = max((self.kernel_size[1] - 1) // 2, 0)
        pad_w = max((self.kernel_size[2] - 1) // 2, 0)
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
        return self.conv(x)

class VGG3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG3D, self).__init__()
        self.features = nn.Sequential(
            Conv3DSame(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            Conv3DSame(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            Conv3DSame(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2)),  # Strided conv instead of max pooling

            Conv3DSame(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            Conv3DSame(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            Conv3DSame(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),  # Strided conv instead of max pooling

            Conv3DSame(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            Conv3DSame(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            Conv3DSame(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            Conv3DSame(256, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2)),  # Strided conv instead of max pooling

            Conv3DSame(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2)),  # Strided conv instead of max pooling

            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3DSame(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2)),  # Strided conv instead of max pooling
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def vgg3d(in_channels, num_classes):
    return VGG3D(in_channels, num_classes)


import torch
import torch.nn as nn
import torchvision.models as models
import timm
from collections import OrderedDict

# ResNeXt3D

class ResNeXt3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNeXt3D, self).__init__()
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1 = nn.BatchNorm3d(64)
        self.model.relu = nn.ReLU(inplace=True)
        self.model.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self._replace_2d_with_3d()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def _replace_2d_with_3d(self):
        replace_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv3d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=(module.kernel_size[0], module.kernel_size[0], module.kernel_size[0]),
                    stride=(module.stride[0], module.stride[0], module.stride[0]),
                    padding=(module.padding[0], module.padding[0], module.padding[0]),
                    bias=module.bias is not None
                )
                replace_dict[name] = new_module
            elif isinstance(module, nn.BatchNorm2d):
                new_module = nn.BatchNorm3d(module.num_features)
                replace_dict[name] = new_module
            elif isinstance(module, nn.MaxPool2d):
                new_module = nn.MaxPool3d(kernel_size=(module.kernel_size, module.kernel_size, module.kernel_size), 
                                          stride=(module.stride, module.stride, module.stride), 
                                          padding=(module.padding, module.padding, module.padding))
                replace_dict[name] = new_module
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_module = nn.AdaptiveAvgPool3d(output_size=(module.output_size[0], module.output_size[0], module.output_size[0]))
                replace_dict[name] = new_module

        # Apply the replacements
        for name, new_module in replace_dict.items():
            self._set_module_by_name(name, new_module)

    def _set_module_by_name(self, name, new_module):
        components = name.split('.')
        module = self.model
        for comp in components[:-1]:
            module = getattr(module, comp)
        setattr(module, components[-1], new_module)

    def forward(self, x):
        return self.model(x)

def resnext3d(in_channels, num_classes):
    return ResNeXt3D(in_channels, num_classes)


# ViT3D
import torch
import torch.nn as nn
from torch.nn import Conv3d, Dropout, LayerNorm
from torch.utils.data import DataLoader, Subset
import copy

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        decoder_channels = config['decoder_channels']
        encoder_channels = config['encoder_channels']
        self.down_num = config['down_num']
        self.inc = Conv3dReLU(n_channels, encoder_channels[0], kernel_size=3, padding=1)
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]

    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        feats = self.down2(x2)
        features.append(feats)
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]

class ViTVNet(nn.Module):
    def __init__(self, config, img_size=(16, 128, 128), vis=False):
        super(ViTVNet, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, img_size)
        self.reg_head = RegistrationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x):
        source = x[:,0:1,:,:]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        #flow = self.integrate(flow)
        out = self.spatial_trans(source, flow)
        return out, flow

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["transformer"]["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.attention_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn_norm = LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config["transformer"]["num_heads"]
        self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config["hidden_size"], self.all_head_size)
        self.key = Linear(config["hidden_size"], self.all_head_size)
        self.value = Linear(config["hidden_size"], self.all_head_size)

        self.out = Linear(config["hidden_size"], config["hidden_size"])
        self.attn_dropout = Dropout(config["transformer"]["attention_dropout_rate"])
        self.proj_dropout = Dropout(config["transformer"]["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config["hidden_size"], config["transformer"]["mlp_dim"])
        self.fc2 = Linear(config["transformer"]["mlp_dim"], config["hidden_size"])
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config["transformer"]["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        down_factor = config["down_factor"]
        patch_size = _triple(config["patches"]["size"])
        n_patches = int((img_size[0] // 2 ** down_factor // patch_size[0]) *
                        (img_size[1] // 2 ** down_factor // patch_size[1]) *
                        (img_size[2] // 2 ** down_factor // patch_size[2]))
        self.hybrid_model = CNNEncoder(config, n_channels=2)
        in_channels = config["encoder_channels"][-1]
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config["hidden_size"],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config["hidden_size"]))
        self.dropout = Dropout(config["transformer"]["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings[:, :x.size(1), :]
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, bn, relu)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv3dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config["down_factor"]
        head_channels = config["conv_first_channel"]
        self.img_size = img_size
        self.conv_more = Conv3dReLU(config["hidden_size"], head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config["decoder_channels"]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config["patches"]["size"])
        skip_channels = self.config["skip_channels"]
        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        l, h, w = (self.img_size[0] // 2 ** self.down_factor // self.patch_size[0], 
                   self.img_size[1] // 2 ** self.down_factor // self.patch_size[1], 
                   self.img_size[2] // 2 ** self.down_factor // self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config["n_skip"]) else None
                if skip is not None:
                    print(f"Shape of skip {i}: {skip.shape}")
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            print(f"Shape of x after decoder block {i}: {x.shape}")
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(dist.Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

# Configuration for ViTVNet
config = {
    "hidden_size": 768,
    "transformer": {
        "num_heads": 12,
        "attention_dropout_rate": 0.1,
        "num_layers": 12,
        "mlp_dim": 3072,
        "dropout_rate": 0.1,
    },
    "down_factor": 4,
    "conv_first_channel": 64,
    "decoder_channels": [128, 64, 32],
    "encoder_channels": [64, 128, 256],
    "skip_channels": [256, 128, 64],
    "n_skip": 3,
    "patches": {"size": [2, 4, 4]},
    "n_dims": 3,
    "down_num": 2
}

# Initialize the model with configuration
def create_vit_vnet():
    return ViTVNet(config)
