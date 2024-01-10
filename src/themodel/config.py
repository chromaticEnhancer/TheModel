import torch
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BATCH_SIZE: int = 1

    IMAGE_WIDTH: int = 112
    IMAGE_HEIGHT: int = 112

    LEARNING_RATE: float = 1e-5
    LAMBDA_CYCLE: int = 10
    NUM_WORKERS: int = 4
    NUM_EPOCHS: int = 10

    TRAIN_BW_MANGA_PATH: str = './data/train/bw'
    TRAIN_COLOR_MANGA_PATH: str = './data/train/color'

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    LOAD_CHECKPOINTS: bool = False
    SAVE_CHECKPOINTS: bool = False
    CHECKPOINTS_FOLDER: str = "./checkpoints"

    OUTPUT_FOLDER: str = "./output"
    OUTPUT_LOSS: str = OUTPUT_FOLDER + "/losses"
    


@lru_cache(maxsize=1)
def setting_func() -> Settings:
    return Settings() #type:ignore

settings = setting_func()