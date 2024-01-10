from enum import Enum
import os
from typing import Literal
import torch

from themodel.config import settings


class CheckpointTypes(str, Enum):
    COLOR_GENERATOR = "color_generator"
    BW_GENERATOR = "bw_generator"
    COLOR_DISC = "color_discriminator"
    BW_DISC = "bw_discriminator"

def save_model(model, optimizer, checkpoint_type: Literal[CheckpointTypes.BW_GENERATOR, CheckpointTypes.COLOR_GENERATOR, CheckpointTypes.COLOR_DISC, CheckpointTypes.BW_DISC]) -> None:
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value, ".pth.tar"))


def load_model(checkpoint_type: Literal[CheckpointTypes.BW_GENERATOR, CheckpointTypes.COLOR_GENERATOR, CheckpointTypes.COLOR_DISC, CheckpointTypes.BW_DISC], model, optimizer, lr = None):
    checkpoint = torch.load(
        f=os.path.join(settings.CHECKPOINTS_FOLDER, checkpoint_type.value, ".pth.tar"), map_location=settings.DEVICE
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr