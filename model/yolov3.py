import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), (stride, stride),
                              (padding, padding), bias=not use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        if self.use_bn:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            ConvBlock(in_channels, in_channels // 2, kernel_size=1),
            ConvBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.bottleneck(x) + x


class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.last_channel = (num_classes + 5) * 3

        # backbone Darknet53
        self.block0 = ConvBlock(3, 32, kernel_size=3, padding=1)
        self.block1 = self.__make_residual_layer(32, 1)
        self.block2 = self.__make_residual_layer(64, 2)
        self.block3 = self.__make_residual_layer(128, 8)
        self.block4 = self.__make_residual_layer(256, 8)
        self.block5 = self.__make_residual_layer(512, 4)

        # feature_maps 1
        self.fm1_block1, self.fm1_block2 = self.__make_yolo_layer(1024, 512, self.last_channel)

        # feature_maps 2
        self.conv_upsample1 = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )
        self.fm2_block1, self.fm2_block2 = self.__make_yolo_layer(768, 256, self.last_channel)

        # feature_maps 3
        self.conv_upsample2 = nn.Sequential(
            ConvBlock(256, 128, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )
        self.fm3_block1, self.fm3_block2 = self.__make_yolo_layer(384, 128, self.last_channel)

    @staticmethod
    def __make_residual_layer(in_channels, repeat):
        layer = list()
        layer.append(('conv_block', ConvBlock(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1)))
        for i in range(repeat):
            layer.append((f'residual_{i}', Bottleneck(in_channels * 2)))
        return nn.Sequential(OrderedDict(layer))

    @staticmethod
    def __make_yolo_layer(in_channels, next_channels, last_channels):
        return (
            nn.Sequential(
                ConvBlock(in_channels, next_channels, kernel_size=1),
                ConvBlock(next_channels, next_channels * 2, kernel_size=3, padding=1),
                ConvBlock(next_channels * 2, next_channels, kernel_size=1),
                ConvBlock(next_channels, next_channels * 2, kernel_size=3, padding=1),
                ConvBlock(next_channels * 2, next_channels, kernel_size=1)
            ),
            nn.Sequential(
                ConvBlock(next_channels, next_channels * 2, kernel_size=3, padding=1),
                ConvBlock(next_channels * 2, last_channels, kernel_size=1, use_bn=False)
            )
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        route_out1 = self.block3(out)
        route_out2 = self.block4(route_out1)
        out = self.block5(route_out2)

        route_out3 = self.fm1_block1(out)
        feature_map_1 = self.fm1_block2(route_out3)

        out = torch.cat((self.conv_upsample1(route_out3), route_out2), dim=1)
        route_out4 = self.fm2_block1(out)
        feature_map_2 = self.fm2_block2(route_out4)

        out = torch.cat((self.conv_upsample2(route_out4), route_out1), dim=1)
        feature_map_3 = self.fm3_block2(self.fm3_block1(out))

        def scale(inputs):
            return inputs.reshape(inputs.size()[0], 3,
                                  self.num_classes + 5,
                                  inputs.size()[2], inputs.size()[3]
                                  ).permute(0, 1, 3, 4, 2).contiguous()

        return scale(feature_map_1), scale(feature_map_2), scale(feature_map_3)
