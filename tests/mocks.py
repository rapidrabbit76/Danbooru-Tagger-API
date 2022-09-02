import typing as T
from functools import lru_cache
from service_streamer import ManagedModel, Streamer
import torch


class EmbeddingSteamerMock(ManagedModel):
    def init_model(self):
        ...

    @torch.inference_mode()
    def predict(self, images: T.List[torch.Tensor]):
        return [torch.rand([512])] * len(images)


class TaggerSteamerMock(ManagedModel):
    def init_model(self):
        ...

    @torch.inference_mode()
    def predict(self, images: T.List[torch.Tensor]):
        return [torch.rand([6000])] * len(images)


@lru_cache(maxsize=1)
def get_embedding_streamer_mock():
    streamer = Streamer(
        EmbeddingSteamerMock,
        batch_size=2,
        max_latency=0.01,
        worker_num=1,
        cuda_devices=[0],
    )
    return streamer


@lru_cache(maxsize=1)
def get_tagger_streamer_mock():
    streamer = Streamer(
        TaggerSteamerMock,
        batch_size=2,
        max_latency=0.01,
        worker_num=1,
        cuda_devices=[0],
    )
    return streamer
