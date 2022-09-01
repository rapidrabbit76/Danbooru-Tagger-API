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
from ..schema import Tag
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
    def predict_embedding(
        self, images: T.List[Image.Image]
    ) -> T.List[T.List[float]]:
        images = [self.preprocessing(image) for image in images]
        outputs = self.embedding_streamer.predict(images)
        outputs = [output.tolist() for output in outputs]
        return outputs

    @torch.inference_mode()
    def predict_score(
        self,
        images: T.List[Image.Image],
    ) -> T.List[float]:
        images = [self.preprocessing(image) for image in images]
        embeddings = self.embedding_streamer.predict(images)
        embeddings = [
            torch.unsqueeze(embedding, dim=0) for embedding in embeddings
        ]
        preds = self.tagging_streamer.predict(embeddings)
        output = [pred.tolist() for pred in preds]
        return output

    @torch.inference_mode()
    def predict_tags(
        self,
        images: T.List[Image.Image],
        threshold: float,
    ) -> T.List[T.List[Tag]]:
        images = [self.preprocessing(image) for image in images]
        embeddings = self.embedding_streamer.predict(images)
        embeddings = [
            torch.unsqueeze(embedding, dim=0) for embedding in embeddings
        ]
        preds = self.tagging_streamer.predict(embeddings)
        tags = [self.postprocessing(pred.numpy(), threshold) for pred in preds]
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
    def postprocessing(score, threshold) -> Tag:
        tmp = score[score > threshold]
        index = score.argsort()[::-1]
        tags = [
            Tag(
                name=CLASSES[i],
                score=score[i],
            )
            for i in index[: len(tmp)]
        ]
        return tags
