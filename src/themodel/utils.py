import os
from enum import Enum
from typing import Literal

import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from themodel.config import settings


class CheckpointTypes(str, Enum):
    COLOR_GENERATOR = "color_generator"
    BW_GENERATOR = "bw_generator"
    COLOR_DISC = "color_discriminator"
    BW_DISC = "bw_discriminator"


def save_model(
    model,
    optimizer,
    checkpoint_type: Literal[
        CheckpointTypes.BW_GENERATOR,
        CheckpointTypes.COLOR_GENERATOR,
        CheckpointTypes.COLOR_DISC,
        CheckpointTypes.BW_DISC,
    ],
) -> None:
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}

    torch.save(
        checkpoint,
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value, ".pth.tar"),
    )


def load_model(
    checkpoint_type: Literal[
        CheckpointTypes.BW_GENERATOR,
        CheckpointTypes.COLOR_GENERATOR,
        CheckpointTypes.COLOR_DISC,
        CheckpointTypes.BW_DISC,
    ],
    model,
    optimizer,
    lr=None,
):
    checkpoint = torch.load(
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value, ".pth.tar"),
        map_location=settings.DEVICE,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def plot_loss(loss_list: list, loss_name: str, title: str) -> None:
    
    plt.figure(figsize=(10,5))

    plt.plot(loss_list)
    

    plt.xlabel('Epochs')
    plt.ylabel(loss_name)
    plt.title(title)

    
    plt.savefig(settings.OUTPUT_LOSS + "/" + title + ".png")
    plt.close()


def make_deterministic():
    seed=0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image(is_color: bool = True):
    mean = (settings.DATASET_MEAN_R_CO, settings.DATASET_MEAN_G_CO, settings.DATASET_MEAN_B_CO)
    std = (settings.DATASET_STD_R_CO, settings.DATASET_STD_G_CO, settings.DATASET_STD_B_CO)

    if not is_color:
        mean = (settings.DATASET_MEAN_R_BW, settings.DATASET_MEAN_G_BW, settings.DATASET_MEAN_B_BW)
        std = (settings.DATASET_STD_R_BW, settings.DATASET_STD_G_BW, settings.DATASET_STD_B_BW)

    transform = transforms.Compose(
        [
            transforms.Resize(size=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH), antialias=True),#type:ignore
            transforms.Normalize(
                mean=mean,
                std=std
            ),
            transforms.ToTensor()
        ]
    )

    return transform


def denormalize_image(image: torch.Tensor, is_color: bool = True):
    mean = (settings.DATASET_MEAN_R_CO, settings.DATASET_MEAN_G_CO, settings.DATASET_MEAN_B_CO)
    std = (settings.DATASET_STD_R_CO, settings.DATASET_STD_G_CO, settings.DATASET_STD_B_CO)

    if not is_color:
        mean = (settings.DATASET_MEAN_R_BW, settings.DATASET_MEAN_G_BW, settings.DATASET_MEAN_B_BW)
        std = (settings.DATASET_STD_R_BW, settings.DATASET_STD_G_BW, settings.DATASET_STD_B_BW)

    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    #rescale the image
    image = image.clamp(0, 1)

    return image


def manage_loss(loss_list: list, epoch_no: int)-> list:
    sum = 0
    for i in range(epoch_no, len(loss_list)):
        sum += loss_list[i]
    
    loss_list[epoch_no] = sum / len(loss_list[epoch_no:])

    return loss_list[0:epoch_no + 1]


# TODO: 6 FUNCTIONS :
# TO CALCULATE MEAN, SD ( OF EACH CHANNEL AND OF 2 DATASET (B&W / COLOURED))
# GENERATING TOTAL OF 6 VALUES

    
if __name__ == "__main__":
    loss = [1, 2, 3]
    print(manage_loss(loss, 1))
