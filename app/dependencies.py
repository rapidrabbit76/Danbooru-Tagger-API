from fastapi.logger import logger
import torch
from .managers import (
    get_danbooru_tagger_streamer,
    get_danbooru_embedding_streamer,
)
from .settings import get_settings


logger.info("---------- dependencies init -------------")
env = get_settings()
torch.set_grad_enabled(False)
logger.info("model loaded Start ")
get_danbooru_tagger_streamer()
get_danbooru_embedding_streamer()
logger.info("---------- dependencies init done ----------")
