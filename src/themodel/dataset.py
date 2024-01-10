import os

import torch
import torchvision
from torch.utils.data import Dataset

from themodel import settings


class BWColorMangaDataset(Dataset):
    def __init__(self, bw_manga_path: str, color_manga_path: str) -> None:
        self.bw_root = bw_manga_path
        self.color_root = color_manga_path

        self.bw_images = os.listdir(self.bw_root)
        self.color_images = os.listdir(self.color_root)
        
        self.dataset_length = min(len(self.bw_images), len(self.color_images))

    def __len__(self) -> int:
        return self.dataset_length
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        bw_image = torchvision.io.read_image(path=os.path.join(self.bw_root, self.bw_images[index]), mode=torchvision.io.ImageReadMode.RGB)
        color_image = torchvision.io.read_image(path=os.path.join(self.color_root, self.color_images[index]), mode=torchvision.io.ImageReadMode.RGB)

        transform = torchvision.transforms.Resize(size=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), antialias=True) #type:ignore

        return transform(bw_image), transform(color_image)
    


def test():
    import matplotlib.pyplot as plt

    train_dataset = BWColorMangaDataset(bw_manga_path=settings.TRAIN_BW_MANGA_PATH, color_manga_path=settings.TRAIN_COLOR_MANGA_PATH)
    bw_image, color_image = train_dataset[0]

    # Display the black and white image
    plt.subplot(1, 2, 1)
    plt.imshow(bw_image.permute(1, 2, 0))
    plt.title('Black and White Image')

    # Display the color image
    plt.subplot(1, 2, 2)
    plt.imshow(color_image.permute(1, 2, 0))
    plt.title('Color Image')

    plt.show()
    