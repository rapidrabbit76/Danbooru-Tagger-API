import typing as T
from functools import lru_cache
from pydantic import BaseSettings


class ModelSettings(BaseSettings):
    BASE_URL = "http://localhost:8080"


class Settings(
    ModelSettings,
):
    pass


@lru_cache()
def get_settings():
    setting = Settings()
    return setting
