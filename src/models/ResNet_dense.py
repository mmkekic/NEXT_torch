

import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, mom=0.1):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, mom=0.1):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes, momentum=mom)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes, momentum=mom)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, momentum=mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Feature_extr(nn.Module):
    def __init__(self,
                 n_initial_filters=8,
                 mom=0.1):
        super().__init__()
        block = BasicBlock
        layers = [1, 1, 1, 2]
        block_inplanes = [16, 32, 64, 128]
        self.in_planes = n_initial_filters
        self.conv0 = nn.Conv3d(1,
                               self.in_planes,
                               kernel_size=(3, 3, 5),
                               stride= 1,
                               padding=0,
                               bias=False)
        self.bn0 = nn.BatchNorm3d(self.in_planes, momentum=mom)
        self.conv1 = nn.Conv3d(self.in_planes,
                               self.in_planes,
                               kernel_size=(5, 5, 15),
                               stride=(2, 2, 4),
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes, momentum=mom)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], mom=mom)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       stride=2,
                                       mom=mom)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       stride=2, 
                                       mom=mom)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       stride=2, 
                                       mom=mom)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.n_final_filters = block_inplanes[3] * block.expansion

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     if isinstance(m, Bottleneck):
        #         nn.init.constant_(m.bn3.weight, 0)
        #     elif isinstance(m, BasicBlock):
        #         nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, mom=0.1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion, momentum=mom))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  mom=mom))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

