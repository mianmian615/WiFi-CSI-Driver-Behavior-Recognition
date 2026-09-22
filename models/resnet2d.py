import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    ResNet中的普通残差块，适用于2D卷积。
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    """
    ResNet中的Bottleneck残差块，适用于2D卷积。
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_shape, hparams, block=Block,  layer_list=[2,2,2,2]):
        super().__init__()
        self.inchannels = input_shape[0]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layer_list[0], planes=64)
        self.layer2 = self._make_layer(block, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block, layer_list[3], planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_outputs = 512 * block.expansion
        
    def _make_layer(self, block, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes * block.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * block.expansion
        for i in range(blocks - 1):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return x

class ResNetLargeBert3(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=55,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNetLargeBert3, self).__init__()
        self.include_top = include_top
        self.in_channel = 128

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(17, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, blocks_num[0])
        self.layer2 = self._make_layer(block, 256, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 512, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(1024 * block.expansion, num_classes)
            # self.fc_bert = nn.Linear(1024 * block.expansion, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            out_bert = torch.flatten(x, 1)
            out = self.fc(out_bert)
        return out, out_bert
# =========================
# ResNet18/ResNet50/ResNet101 别名工厂
# =========================
def ResNet18(input_shape, hparams):
    return ResNet(input_shape, hparams, block=Block, layer_list=[2, 2, 2, 2])

def ResNet18_mutual(input_shape, hparams):
    return ResNetLargeBert3(input_shape, hparams, block=Block, layer_list=[2, 2, 2, 2])

def ResNet34(input_shape, hparams):
    return ResNet(input_shape, hparams, block=Block, layer_list=[3, 4, 6, 3])

def ResNet34_mutual(input_shape, hparams):
    return ResNetLargeBert3(input_shape, hparams, block=Block, layer_list=[3, 4, 6, 3])


def ResNet50(input_shape, hparams):
    return ResNet(input_shape, hparams, block=Bottleneck, layer_list=[3, 4, 6, 3])

def ResNet50_mutual(input_shape, hparams):
    return ResNetLargeBert3(input_shape, hparams, block=Bottleneck, layer_list=[3, 4, 6, 3])

def ResNet101(input_shape, hparams):
    return ResNet(input_shape, hparams, block=Bottleneck, layer_list=[3, 4, 23, 3])

def ResNet101_mutual(input_shape, hparams):
    return ResNetLargeBert3(input_shape, hparams, block=Bottleneck, layer_list=[3, 4, 23, 3])

def ResNet1111(input_shape, hparams):
    """ResNet18结构，4个Block残差层。"""
    return ResNet(input_shape, hparams, block=Block, layer_list=[1, 1, 1, 1])

def ResNet2346(input_shape, hparams):
    """ResNet18结构，4个Block残差层。"""
    return ResNet(input_shape, hparams, block=Block, layer_list=[2, 3, 4, 6])
