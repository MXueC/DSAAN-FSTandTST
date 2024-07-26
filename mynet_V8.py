#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :mynet_V8.py
# @Time      :2023/9/15 9:03
# @Author    :Mengxc



import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    "Basic Block for resnet18 and resnet 34"
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * BasicBlock.expansion, kernel_size=3
                      , padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)

        )


        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion, kernel_size=1
                          , stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# unused
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)  #
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSAAN(nn.Module):
    def __init__(self, block, num_block, long_len, num_classes=2, short_len=3) -> None:
        # block
        super().__init__()
        if not short_len % 2:
            self.short_len = short_len + 1
        else:
            self.short_len = short_len
        if not long_len % 2:
            self.long_len = long_len + 1
        else:
            self.long_len = long_len
        self.in_channels = 64
        self.pw1_outchannels = 64
        self.pw2_outchannels = 64
        self.pathway1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pw1_conv2 = self._make_layer(block, 128, num_block[0], 1)
        self.pw1_conv3 = self._make_layer(block, self.pw2_outchannels, num_block[1], 1)


        self.pw2_conv1 = nn.Sequential(
            nn.Conv2d(self.short_len, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.pw2_conv2 = self._make_layer(block, 64, num_block[0], 1)
        self.pw2_conv3 = self._make_layer(block, self.pw2_outchannels, num_block[1], 1)

        self.conv4 = nn.Conv2d(64,2,kernel_size=3) #
        self.fuse = nn.Conv2d(self.pw1_outchannels + self.pw2_outchannels, 2, kernel_size=3, padding=1, bias=False)
        self.dw = depthwise_separable_conv(nin=self.long_len, nout=64)   # unused
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x_pw1 = x[:, -1, :, :]  # ( batchsize, 112,112)
        x_pw1 = torch.unsqueeze(x_pw1, dim=1)  # (batchsize,1 ,112,112)
        x1 = self.pathway1(x_pw1)  # (batchsize, 64,112,112)
        x1 = self.pw1_conv2(x1)  # (batchsize,128,112,112)
        x1 = self.pw1_conv3(x1)  # (batchsize,64,112,112)
        out1 = self.avg_pool(self.conv4(x1))
        out1 = out1.view(out1.size(0), -1)


        x2 = x[:, (self.long_len // 2 - self.short_len // 2):(self.long_len // 2 + self.short_len // 2 + 1), :,:]
        # (batchsize, short_len, 112,112)
        x2 = self.pw2_conv1(x2)  # (batchsize, 64, 112,112)
        x2 = self.pw2_conv2(x2)  # (batchsize, 64, 112,112)
        x2 = self.pw2_conv3(x2)  # (batchsize, 64, 112,112)
        out2 = self.avg_pool(self.conv4(x2))
        out2 = out2.view(out2.size(0), -1)

        # x = self.conv4(x)
        # x = self.conv5(x)
        x = torch.cat([x1, x2], dim=1)  # concat (batchsize, 128, 112,112)
        x = self.fuse(x)  # (batchsize, 2, 112,112)

        x = self.avg_pool(x)  # (batchsize, 2, 1,1)
        x = x.view(x.size(0), -1)  # (batchsize, 2)
        # x = self.fc(x)
        return x,out1,out2


def mynet(long_len, short_len):
    return DSAAN(BasicBlock, [2, 2, 2, 2], long_len=long_len, short_len=short_len)
