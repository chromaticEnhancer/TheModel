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

    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512

    LEARNING_RATE: float = 1e-5
    LAMBDA_CYCLE: int = 10
    NUM_WORKERS: int = 2
    NUM_EPOCHS: int = 100

    TRAIN_BW_MANGA_PATH: str = "./data/train/bw"
    TRAIN_COLOR_MANGA_PATH: str = "./data/train/color"

    # DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE: str = "cuda"

    LOAD_CHECKPOINTS: bool = False
    SAVE_CHECKPOINTS: bool = False
    CHECKPOINTS_FOLDER: str = "./checkpoints"

    OUTPUT_FOLDER: str = "./output"
    


@lru_cache(maxsize=1)
def setting_func() -> Settings:
    return Settings()  # type:ignore


settings = setting_func()
