import torch
import torchvision

def get_mean_std(image: torch.Tensor):
    """
    Calculate the mean and standard deviation of an image tensor

    Args:
        img_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing two lists - the means and standard deviations of the image channels.

    """
    means = []
    stds = []
    
    for i in range(image.shape[0]):
        means.append(image[i, :, :].float().mean().item())
        stds.append(image[i, :, :].float().std().item())

    return means, stds

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean, std = get_mean_std(tensor)
    
    denorm = torchvision.transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std]
    )
    out = denorm(tensor)
    out = (out - out.min()) / (out.max() - out.min()) * 255

    return out