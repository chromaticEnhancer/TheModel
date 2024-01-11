import os

import torch
import torchvision
from torch.utils.data import Dataset

from themodel import settings
from themodel.utils import normalize_image, denormalize_image

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
            mode=torchvision.io.ImageReadMode.RGB,
        )
        color_image = torchvision.io.read_image(
            path=os.path.join(self.color_root, self.color_images[index]),
            mode=torchvision.io.ImageReadMode.RGB,
        )

        normalise_color = normalize_image(is_color=True)
        normalise_bw = normalize_image(is_color=False)
        
        return normalise_bw(bw_image), normalise_color(color_image)

