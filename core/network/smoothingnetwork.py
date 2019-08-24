import torch
from torch import nn

class SmoothingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_reduce_half = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.resnets = []
        for dilation_power in range(1,5):


    def forward(self, image):
        return image


class ResNet(nn.Module):
    def __init__(self, dilation, padding, in_channel=64, out_channel=64, stride=1):
        super().__init__()

        self.padding = padding
        self.dilation = dilation
        self.W = None if in_channel == out_channel else nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                                                  kernel_size=1)

        norm_layer = nn.BatchNorm2d
        self.conv1 = self.conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(out_channel, out_channel, stride)
        self.bn2 = norm_layer(out_channel)



    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=self.padding, dilation=self.dilation,bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.W:
            identity = self.W(identity)

        out += identity
        out = self.relu(out)

        return out
