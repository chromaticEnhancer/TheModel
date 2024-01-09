from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


class LocalFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = self.__conv_lay_w_leak(
            in_channel=3, out_channel=32, is_first_layer=True
        )
        self.layer2 = self.__conv_lay_w_leak(in_channel=32, out_channel=64)
        self.layer3 = self.__conv_lay_w_leak(in_channel=64, out_channel=92)
        self.layer4 = self.__conv_lay_w_leak(in_channel=92, out_channel=128)
        self.layer5 = self.__conv_lay_w_leak(in_channel=128, out_channel=256)

    def __conv_lay_w_leak(
        self, in_channel, out_channel, is_first_layer: bool = False
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

    def forward(self, image) -> Tuple[nn.Sequential, nn.Sequential]:
        """
        Image should have 3 channels.
        aux_out(1st output), xo(2nd output)
        """
        x0 = self.layer1(image)
        aux_out = self.layer2(x0)
        aux_out = self.layer3(aux_out)
        aux_out = self.layer4(aux_out)
        aux_out = self.layer5(aux_out)

        return aux_out, x0





def test():
    feature = LocalFeatureExtractor()
    img = torch.randn(1, 3, 512, 512).to('cuda')
    model = feature.to('cuda')
    aux, x0 = model(img)

    print(f'aux shape, {aux.shape}, xo shape, {x0.shape}')


if __name__ == "__main__":
    test()