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
        # Adjust the initial conv layer to handle 3D input
        self.model.conv_stem = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Adjust batch normalization layers
        self.model.bn1 = nn.BatchNorm3d(32)
        # Replace any 2D layers to handle 3D
        self._replace_2d_with_3d()
        # Adjust classifier for the number of output classes
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
    def __init__(self, num_classes=1000):
        super(EfficientNet3D, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', Conv3dSame(3, 32, kernel_size=3, stride=2)),
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
    model = EfficientNet3D(num_classes=num_classes)
    return model

def efficientnet3d_b7(in_channels, num_classes):
    model = EfficientNet3D(num_classes=num_classes)
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

class ViT3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ViT3D, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.patch_embed.proj = nn.Conv3d(in_channels, self.model.embed_dim, kernel_size=(16, 16, 16), stride=(16, 16, 16))
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
        for block in self.model.blocks:
            block.attn.qkv = nn.Linear(block.attn.qkv.in_features, block.attn.qkv.out_features)
            block.attn.proj = nn.Linear(block.attn.proj.in_features, block.attn.proj.out_features)
            block.mlp.fc1 = nn.Linear(block.mlp.fc1.in_features, block.mlp.fc1.out_features)
            block.mlp.fc2 = nn.Linear(block.mlp.fc2.in_features, block.mlp.fc2.out_features)

    def forward(self, x):
        B, C, D, H, W = x.shape  # Expecting a 5D input
        x = self.model.patch_embed.proj(x).flatten(2).transpose(1, 2)  # Adjust the patch embedding layer
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)
        return self.model.head(x[:, 0])

def vit3d(in_channels, num_classes):
    return ViT3D(in_channels, num_classes)


def list_available_timm_models():
    model_names = timm.list_models(pretrained=True)
    print("Available models in timm:")
    for name in model_names:
        print(name)

if __name__ == "__main__":
    list_available_timm_models()
