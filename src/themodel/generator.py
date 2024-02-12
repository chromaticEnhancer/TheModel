import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, input_channel: int):
        super().__init__()

        self.layers = nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, padding=0, bias=True),
                        nn.InstanceNorm2d(input_channel),
                        nn.ReLU(True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=3, padding=0, bias=True),
                        nn.InstanceNorm2d(input_channel)
                    )

    def forward(self, x):
        return x + self.layers(x)
    
class ResNet9Generator(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.blocks = 9
        self.features = 64
        self.layers = [
            nn.Conv2d(in_channels=input_channels, out_channels=self.features, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(self.features),
            nn.ReLU(True)
        ]

        nDownSampling = 2
        for i in range(nDownSampling):
            mult = 2 ** i
            self.layers += [
                nn.Conv2d(self.features * mult, self.features * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(self.features * mult * 2),
                nn.ReLU(True)
            ]


        mult = 2 ** nDownSampling
        for i in range(self.blocks):
            self.layers += [
                ResNetBlock(self.features * mult)
            ]
        
        for i in range(nDownSampling):
            mult = 2 ** (nDownSampling - i)
            self.layers += [
                nn.ConvTranspose2d(in_channels=self.features * mult, out_channels=int(self.features * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(self.features * mult / 2)),
                nn.ReLU(True)
            ]

        self.layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.features, output_channels, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    model = ResNet9Generator(input_channels=3, output_channels=3)
    input = torch.randn(1, 3, 280, 280)
    out = model(input)
    print(out.shape)