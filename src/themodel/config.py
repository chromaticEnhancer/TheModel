from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BATCH_SIZE: int = 1

    IMAGE_WIDTH: int = 112
    IMAGE_HEIGHT: int = 112

    LEARNING_RATE: float = 1e-5
    LAMBDA_IDENTITY: float = 0.0
    LAMBDA_CYCLE: int = 10
    NUM_WORKERS: int = 4
    NUM_EPOCHS: int = 10

    TRAIN_BW_MANGA_PATH: str = './data/train/bw'
    TRAIN_COLOR_MANGA_PATH: str = './data/train/color'
    


@lru_cache(maxsize=1)
def setting_func() -> Settings:
    return Settings() #type:ignore

settings = setting_func()