import torch.nn as nn

class PatchGan(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()

        self.features = 64
        self.numLayers = 3
        self.kernelSize = 4

        self.layers = [
            nn.Conv2d(in_channels=input_channels, out_channels=self.features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        inputChannelMulti = 1
        inputChannelMultiPrev = 1
        for i in range(1, self.numLayers):
            inputChannelMultiPrev = inputChannelMulti
            inputChannelMulti = min(2 ** i, 8)
            self.layers += [
                nn.Conv2d(self.features * inputChannelMultiPrev, self.features * inputChannelMulti, kernel_size=self.kernelSize, stride=1, padding=1, bias=True),
                nn.InstanceNorm2d(self.features * inputChannelMulti),
                nn.LeakyReLU(0.2, True)
            ]

        self.layers += [
            nn.Conv2d(in_channels=self.features * inputChannelMulti, out_channels=1, kernel_size=self.kernelSize, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    from themodel.generator import ResNet9Generator

    gen = ResNet9Generator(3, 3)
    dis = PatchGan(3)

    input = torch.randn(1, 3, 212, 212)
    outgen = gen(input)
    print('Generator ouput', outgen.shape)
    outdisc = dis(outgen)

    print('Discriminator Outpu', outdisc.shape)
    