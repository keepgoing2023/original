import torch
import torch.nn as nn
import torch.nn.functional as F
# (B, 512, 7, 7)
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class Bridge(nn.Module):
    def __init__(self, channel=1):
        super(Bridge, self).__init__()
        self.fuse4 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth4 = DSConv3x3(channel, channel, stride=1, dilation=1)  # 96channel-> 96channel

        self.d_fuse4 = convbnrelu(channel, channel, k=1, s=1, p=0, relu=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d_smooth4 = DSConv3x3(channel, channel, stride=1, dilation=1)  # 96channel-> 96channel

        # self.fuse3 = convbnrelu(channel3, channel3, k=1, s=1, p=0, relu=True)
        # self.smooth3 = DSConv3x3(channel3, channel4, stride=1, dilation=1)  # 32channel-> 96channel
        # self.ChannelCorrelation = CCorrM(channel4, 32)

    def forward(self, rgb, depth, channel):  # x4:96*18*18 k4:96*5*5; x3:32*36*36 k3:32*5*5
        B, C, H, W = rgb.size()
        # k4 = k4.view(C4, 1, H4, W4)
        # k3 = k3.view(C3, 1, H3, W3)
        rgb_fea = rgb.clone()
        x4_r1 = F.conv2d(rgb_fea, stride=1, padding=2, dilation=1, groups=channel)  # 卷积核通常具有四个维度：[out_channels, in_channels, kernel_height, kernel_width]
        x4_r2 = F.conv2d(rgb_fea, stride=1, padding=4, dilation=2, groups=channel)
        x4_r3 = F.conv2d(rgb_fea, stride=1, padding=6, dilation=3, groups=channel)
        rgb_new = x4_r1 + x4_r2 + x4_r3
        # Pconv
        rgb_all = self.fuse4(rgb_new)
        rgb_smooth = self.smooth4(rgb_all)

        dep_fea = depth.clone()
        x4_d1 = F.conv2d(dep_fea, stride=1, padding=2, dilation=1, groups=channel)
        x4_d2 = F.conv2d(dep_fea, stride=1, padding=4, dilation=2, groups=channel)
        x4_d3 = F.conv2d(dep_fea, stride=1, padding=6, dilation=3, groups=channel)
        dep_new = x4_d1 + x4_d2 + x4_d3
        # Pconv
        dep_all = self.d_fuse4(dep_new)
        dep_smooth = self.d_smooth4(dep_all)

        # Channel-wise Correlation
        # x3_out, x4_out = self.ChannelCorrelation(x3_smooth, x4_smooth)

        return rgb_smooth, dep_smooth  # (96*2)*32*32

