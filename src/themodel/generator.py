import torch
import torch.nn as nn

from themodel.components import UNetEncoder
from themodel.components import UNetDecoder
from themodel.components import LocalFeatureExtractor


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.feature = LocalFeatureExtractor()

        self.exit = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, image):
        aux_out, x0 = self.feature(image)
        x1, x2, x3, x4 = self.encoder(image)

        out = self.decoder.layer4(torch.cat([x4, aux_out], 1))

        x = self.decoder.layer3(torch.cat([out, x3], 1))

        x = self.decoder.layer2(torch.cat([x, x2, x1], 1))

        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))

        return x


# 512 512
# 112 112
