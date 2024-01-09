import math
from typing import Any

import torch
import torch.nn as nn

#a random comment

class SELayer(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, 1),
            nn.Conv2d(in_channels // 16, in_channels, 1, 1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, image):
        return image * self.layer(image)


class SEBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cardinality: int,
        stride: int = 1,
        downsample: Any = None,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.stride = stride
        self.selayer = SELayer(out_channels * 4)

        self.layer_first_half = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * 2,
                out_channels * 2,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=cardinality,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            SELayer(out_channels * 4),
        )

        self.layer_sec_half = nn.ReLU(inplace=True)

    def forward(self, input):
        res = input
        out = self.layer_first_half(input)

        if self.downsample is not None:
            res = self.downsample(input)

        out += res
        return self.layer_sec_half(out)


class UNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cardinality = 32
        self.inplanes = 64
        self.input_channels = 3  #code ma 1 channel xa

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer2 = self.__make_layer(out_channels=64, no_of_bottlenecks=3)
        self.layer3 = self.__make_layer(out_channels=128, no_of_bottlenecks=4, stride=2)
        self.layer4 = self.__make_layer(out_channels=256, no_of_bottlenecks=6, stride=2)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __downsample(
        self, in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    def __make_layer(
        self, out_channels: int, no_of_bottlenecks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != out_channels * 4:
            downsample = self.__downsample(
                in_channels=self.inplanes, out_channels=out_channels * 4, stride=stride
            )

        layers = []

        layers.append(
            SEBlock(
                in_channels=self.inplanes,
                out_channels=out_channels,
                cardinality=self.cardinality,
                stride=stride,
                downsample=downsample,
            )
        )

        self.inplanes = out_channels * 4

        for _ in range(1, no_of_bottlenecks):
            layers.append(
                SEBlock(
                    in_channels=self.inplanes,
                    out_channels=out_channels,
                    cardinality=self.cardinality,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, image) -> tuple[Any, Any, Any, Any]:
        x1 = self.layer1(image)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


class LocalFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = self.__conv_lay_w_leak(
            in_channel=3, out_channel=32, is_first_layer=True
        )
        self.layer2 = self.__conv_lay_w_leak(in_channel=32, out_channel=64)
        self.layer3 = self.__conv_lay_w_leak(in_channel=64, out_channel=92)
        self.layer4 = self.__conv_lay_w_leak(in_channel=92, out_channel=128)


    def __conv_lay_w_leak(
        self, in_channel:int , out_channel:int , is_first_layer: bool = False
    ) -> nn.Sequential:
        stride = 2
        if is_first_layer:
            stride = 1

        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image) -> tuple[Any, Any]:
        """
        Image should have 3 channels.
        aux_out(1st output), xo(2nd output)
        """
        x0 = self.layer1(image)
        aux_out = self.layer2(x0)
        aux_out = self.layer3(aux_out)
        aux_out = self.layer4(aux_out)

        return aux_out, x0


def __test():
    image = torch.randn(1, 3, 512, 512).to('cuda')
   
    encoder = UNetEncoder()
    feature = LocalFeatureExtractor()

    model1 = encoder.to('cuda')
    model2 = feature.to('cuda')

    out1 = model1(image)
    out2 = model2(image)
    print("encoder out", end=' ')
    print(out1[3].shape)

    print("feature output", end=' ')
    print(out2[1].shape)

    torch.cat([out1[3], out2[0]], 1)


if __name__ == "__main__":
    __test()