import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

#a random comment

class SEskipConnection(nn.Module):
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


class SEBlockEncoderSide(nn.Module):
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
        self.selayer = SEskipConnection(out_channels * 4)

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
            SEskipConnection(out_channels * 4),
        )

        self.layer_sec_half = nn.ReLU(inplace=True)

    def forward(self, input):
        res = input
        out = self.layer_first_half(input)

        if self.downsample is not None:
            res = self.downsample(input)

        out += res
        return self.layer_sec_half(out)


class SEBlockDecoderSide(nn.Module):
    def __init__(self, input_channels: int, out_channels: int, stride: int = 1, cardinality: int=32, dilate:int = 1 ) -> None:
        """
        input_channels and output_channels must be divisible by cardinality
        """
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size= 2 + stride, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            SEskipConnection(in_channels=out_channels)
        )

        self.conditional_layer = None
        if stride != 1:
            self.conditional_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, image):
        out = self.layer1(image)

        if self.conditional_layer:
            out = self.conditional_layer(out)

        
        return image + out


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
            SEBlockEncoderSide(
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
                SEBlockEncoderSide(
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


class UNetDecoder(nn.Module):
    """
    We don't have forward method here 
    because output from decoder is also required.
    """
    def __init__(self) -> None:
        super().__init__()


        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1152, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[SEBlockDecoderSide(512, 512, cardinality=32, dilate=1) for _ in range(20)],
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[SEBlockDecoderSide(input_channels=256, out_channels=256, cardinality=32, dilate=1) for _ in range(2)],
            *[SEBlockDecoderSide(input_channels=256, out_channels=256, cardinality=32, dilate=2) for _ in range(2)],
            *[SEBlockDecoderSide(input_channels=256, out_channels=256, cardinality=32, dilate=4) for _ in range(2)],
            SEBlockDecoderSide(input_channels=256, out_channels=256, cardinality=32, dilate=2),
            SEBlockDecoderSide(input_channels=256, out_channels=256, cardinality=32, dilate=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128 + 256 + 64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *[SEBlockDecoderSide(input_channels=128, out_channels=128, cardinality=32, dilate=1) for _ in range(2)],
            *[SEBlockDecoderSide(input_channels=128, out_channels=128, cardinality=32, dilate=2) for _ in range(2)],
            *[SEBlockDecoderSide(input_channels=128, out_channels=128, cardinality=32, dilate=4) for _ in range(2)],
            SEBlockDecoderSide(input_channels=128, out_channels=128, cardinality=32, dilate=2),
            SEBlockDecoderSide(input_channels=128, out_channels=128, cardinality=32, dilate=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64 + 32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            SEBlockDecoderSide(input_channels=64, out_channels=64, cardinality=16, dilate=1),
            SEBlockDecoderSide(input_channels=64, out_channels=64, cardinality=16, dilate=2),
            SEBlockDecoderSide(input_channels=64, out_channels=64, cardinality=16, dilate=4),
            SEBlockDecoderSide(input_channels=64, out_channels=64, cardinality=16, dilate=2),
            SEBlockDecoderSide(input_channels=64, out_channels=64, cardinality=16, dilate=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upscale(self, input_channels: int = 256, out_channels: int = 3) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, out_channels, 3, stride=1, padding=1, output_padding=0),
            nn.Tanh()
        )


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
    image_s = torch.randn(1, 3, 512, 512).to('cuda')



    # extractor = SEBlockDecoderSide(input_channels=3, out_channels=5)
    # model2 = extractor.to('cuda')
    # out2 = model2(image_s)
    # print(out2.shape)
    encoder = UNetEncoder()
    feature = LocalFeatureExtractor()

    model1 = encoder.to('cuda')
    model2 = feature.to('cuda')

    out1 = model1(image)
    out2 = model2(image)
    # print("encoder out", end=' ')
    # print(out1[3].shape)

    print("feature output", end=' ')
    print(out2[1].shape)

    layer3_in=torch.cat([out1[3], out2[0]], 1)
  

 

    

if __name__ == "__main__":
    __test()