import typing as T
from fastapi import (
    Depends,
    File,
    UploadFile,
    HTTPException,
    status,
    Body,
)
from fastapi_restful.cbv import cbv

from fastapi_restful.inferring_router import InferringRouter
from fastapi.logger import logger
from app.settings import get_settings
from app.services import DanbooruService

from PIL import Image
from class_name import CLASSES

router = InferringRouter()
setting = get_settings()


@cbv(router)
class Danbooru:
    svc: DanbooruService = Depends()

    @router.post("/predict/embedding", response_model=T.List[T.List[float]])
    def predict_embedding(
        self,
        images: T.List[UploadFile] = File(...),
    ):
        logger.info("------------- Tagger Start -----------")
        images = [self.imread(image) for image in images]
        output = self.svc.predict_embedding(images)
        logger.info("------------- Tagger Done -----------")
        return output

    @router.post("/predict/score", response_model=T.List[float])
    def predict_score(
        self,
        image: UploadFile = File(...),
    ):
        logger.info("------------- Tagger Start -----------")
        image = self.imread(image)
        output = self.svc.predict_score(image)
        logger.info("------------- Tagger Done -----------")
        return output

    @router.post("/predict/tag", response_model=T.List[str])
    def predict_tag(
        self,
        image: UploadFile = File(...),
        threshold: float = Body(0.2, embed=True),
    ):
        logger.info("------------- Tagger Start -----------")
        image = self.imread(image)
        output = self.svc.predict_tags(image, threshold)
        logger.info("------------- Tagger Done -----------")
        return output

    @router.get("/tags", response_model=T.List[str])
    def get_tags(self):
        return CLASSES

    @staticmethod
    def imread(image):
        try:
            image = Image.open(image.file).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail=f"""{image.filename} is not image file, {e} """,
            )
        return image
