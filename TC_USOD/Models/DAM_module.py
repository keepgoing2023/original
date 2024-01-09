import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
# from .ResNet import *
# from .t2t_vit import *
class ChannelAttention(nn.Module):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y
class CA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        depth = depth.mul(self.sigmoid(out))
        return depth

class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CA_SA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_SA_Enhance, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x_d = self.self_CA_Enhance(rgb, depth)
        sa = self.self_SA_Enhance(x_d)
        depth_enhance = depth.mul(sa)
        return depth_enhance


class DAM_module(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(DAM_module, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x_d = self.self_CA_Enhance(rgb, depth)
        sa = self.self_SA_Enhance(x_d)
        depth_enhance = depth.mul(sa)
        return depth_enhance


