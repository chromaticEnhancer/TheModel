import os
from enum import Enum
from typing import Literal, Optional

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
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value),
    )


def load_model(
    model,
    optimizer,
    checkpoint_type: Literal[
        CheckpointTypes.BW_GENERATOR,
        CheckpointTypes.COLOR_GENERATOR,
        CheckpointTypes.COLOR_DISC,
        CheckpointTypes.BW_DISC,
    ],
    lr=None,
):
    checkpoint = torch.load(
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value),
        map_location=settings.DEVICE,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def load_generator(model, checkpoint_type: Literal[CheckpointTypes.COLOR_GENERATOR] | Literal[CheckpointTypes.BW_GENERATOR]):
    checkpoint = torch.load(
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value),
        map_location=settings.DEVICE,
    )
    
    model.load_state_dict(checkpoint["model"])

def make_deterministic():
    seed=0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_plots(loss1: list, l1_label: str, loss2: Optional[list], l2_label: Optional[str], title: str):
    plt.figure()
    plt.plot(loss1, label=l1_label)
    if loss2:
        plt.plot(loss2, label=l2_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(settings.OUTPUT_FOLDER + f'/{title}.jpg')
    plt.close()

