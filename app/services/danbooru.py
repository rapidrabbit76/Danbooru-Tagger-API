import typing as T
import torch
from fastapi import Depends
from fastapi.logger import logger
from PIL import Image
from torchvision.transforms.functional import (
    normalize,
    resize,
    to_tensor,
)
from ..managers import (
    get_danbooru_embedding_streamer,
    get_danbooru_tagger_streamer,
)
from ..settings import get_settings
from class_name import CLASSES

env = get_settings()


class DanbooruService:
    def __init__(
        self,
        embedding_streamer=Depends(get_danbooru_embedding_streamer),
        tagger_streamer=Depends(get_danbooru_tagger_streamer),
    ):
        logger.info(f"DI: {self.__class__.__name__}")
        self.embedding_streamer = embedding_streamer
        self.tagging_streamer = tagger_streamer

    @torch.inference_mode()
    def predict_embedding(self, image: Image.Image) -> T.List[float]:
        image = self.preprocessing(image)
        output = self.embedding_streamer.predict([image])[0]
        output = output.tolist()
        return output

    @torch.inference_mode()
    def predict_score(self, image: Image.Image) -> T.List[float]:
        image = self.preprocessing(image)
        output = self.embedding_streamer.predict([image])[0]
        output = torch.unsqueeze(output, dim=0)
        output = self.tagging_streamer.predict([output])[0]
        output = output.tolist()
        return output

    @torch.inference_mode()
    def predict_tags(
        self,
        image: Image.Image,
        threshold: float,
    ) -> T.List[str]:
        image = self.preprocessing(image)
        output = self.embedding_streamer.predict([image])[0]
        output = torch.unsqueeze(output, dim=0)
        output = self.tagging_streamer.predict([output])[0]
        output = output.numpy()
        tags = self.postprocessing(output, threshold)
        return tags

    @staticmethod
    def preprocessing(image: Image.Image) -> torch.Tensor:
        image = resize(image, (244, 244))
        image = to_tensor(image)
        image = normalize(
            image,
            mean=[0.7137, 0.6628, 0.6519],
            std=[0.2970, 0.3017, 0.2979],
        )
        return image.unsqueeze(0)

    @staticmethod
    def postprocessing(score, threshold):
        tmp = score[score > threshold]
        index = score.argsort()
        tags = [CLASSES[i] for i in index[: len(tmp)]]
        return tags
