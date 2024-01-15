import torch
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BATCH_SIZE: int = 1

    DATASET_MEAN_R_CO: float = 0.1307
    DATASET_MEAN_G_CO: float = 0.1307
    DATASET_MEAN_B_CO: float = 0.1307

    DATASET_MEAN_R_BW: float = 0.1307
    DATASET_MEAN_G_BW: float = 0.1307
    DATASET_MEAN_B_BW: float = 0.1307

    DATASET_STD_R_CO: float = 0.3081
    DATASET_STD_G_CO: float = 0.3081
    DATASET_STD_B_CO: float = 0.3081

    DATASET_STD_R_BW: float = 0.3081
    DATASET_STD_G_BW: float = 0.3081
    DATASET_STD_B_BW: float = 0.3081

    IMAGE_WIDTH: int = 64
    IMAGE_HEIGHT: int = 64

    LEARNING_RATE: float = 1e-5
    LAMBDA_CYCLE: int = 10
    NUM_WORKERS: int = 2

    NUM_EPOCHS: int = 5
    DECAY_EPOCH: int = 10

    USE_WHITE_COLOR_LOSS: bool = True
    USE_INITIALIZED_WEIGHTS: bool = False

    GENERATOR_LR: float = 1e-4
    DISCRIMINATOR_LR: float = 4e-4

    TRAIN_BW_MANGA_PATH: str = "./data/train/bw"
    TRAIN_COLOR_MANGA_PATH: str = "./data/train/color"
    CBZ_FILES: str = "./data/cbz"

    # DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    LOAD_CHECKPOINTS: bool = False
    SAVE_CHECKPOINTS: bool = False
    CHECKPOINTS_FOLDER: str = "./checkpoints"

    OUTPUT_FOLDER: str = "./output"
    


@lru_cache(maxsize=1)
def setting_func() -> Settings:
    return Settings()  # type:ignore


settings = setting_func()