import os

import torch
import torchvision
from torch.utils.data import Dataset

from themodel import settings
from themodel.normalizer import get_mean_std

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
        "bw_image, color_image"
        
        bw_image = torchvision.io.read_image(
            path=os.path.join(self.bw_root, self.bw_images[index]),
            mode=torchvision.io.ImageReadMode.GRAY,
        )
        color_image = torchvision.io.read_image(
            path=os.path.join(self.color_root, self.color_images[index]),
            mode=torchvision.io.ImageReadMode.RGB,
        )

        resize =  torchvision.transforms.Resize(size=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), antialias=True) #type:ignore
        bw_image = resize(bw_image) / settings.DIVIDE_VALUE
        color_image = resize(color_image) / settings.DIVIDE_VALUE
    
        if settings.NORMALIZE_DATASET:
            mean_bw, std_bw = get_mean_std(bw_image)
            mean_color, std_color = get_mean_std(color_image)

            normalize_bw = torchvision.transforms.Normalize(
                mean=mean_bw,
                std=std_bw,
            )

            normalize_color = torchvision.transforms.Normalize(
                mean=mean_color,
                std=std_color,
            )

            bw_image = normalize_bw(bw_image)
            color_image = normalize_color(color_image)

        bw_image = bw_image.repeat(3, 1, 1)
        
        return bw_image, color_image


if __name__ == "__main__":
    settings.NORMALIZE_DATASET = True

    dataset = BWColorMangaDataset(bw_manga_path=settings.TRAIN_BW_MANGA_PATH, color_manga_path=settings.TRAIN_COLOR_MANGA_PATH)
    bw, color = dataset[0]
    print(bw, color)
    print(bw.shape, color.shape)
