import torch
import torch.nn as nn

#another  random comment below, nn.Modulo
class Discriminator(nn.Modulo):
    def __init__(self, input_channels=6, output_channels=1):
        super().__init__()

        self.lay1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.lay2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.lay3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.lay4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.lay5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        self.lay6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=output_channels,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
        )

        self.model = nn.Sequential(
            self.lay1,
            self.lay2,
            self.lay3,
            self.lay4,
            self.lay5,
            self.lay6,
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)

        return self.model(x)


def test():
    x = torch.randn(1, 3, 512, 512).to("cuda")
    y = torch.randn(1, 3, 512, 512).to("cuda")

    model = Discriminator(input_channels=6, output_channels=1)
    model = model.to("cuda")
    preds = model(x, y)

    assert preds.shape == (1, 1, 62, 62), "Output shape is incorrect"
    print(preds.shape)


if __name__ == "__main__":
    test()
