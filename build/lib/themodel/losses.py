import torch
import torchvision.models as models


def white_color_penalty(actual_image: torch.Tensor, generated_image: torch.Tensor):
    mask = (
        ~((actual_image > 0.85).float().sum(dim=1) == 3)
        .unsqueeze(1)
        .repeat((1, 3, 1, 1))
    ).float()
    white_zones = mask * (generated_image + 1) / 2
    white_penalty = (
        torch.pow(white_zones.sum(dim=1), 2).sum(dim=(1, 2))
        / (mask.sum(dim=(1, 2, 3)) + 1)
    ).mean()

    return white_penalty


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:16])
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = torch.nn.functional.l1_loss(x_vgg, y_vgg)
        return loss