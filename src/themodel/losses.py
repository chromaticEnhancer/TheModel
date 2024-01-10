import torch


def white_color_penalty_loss(actual_image: torch.Tensor, generated_image: torch.Tensor):
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
