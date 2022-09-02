from pydoc import cli
import pytest

from fastapi.testclient import TestClient
from app.main import create_app
from app.managers import (
    get_danbooru_embedding_streamer,
    get_danbooru_tagger_streamer,
)

from .mocks import get_embedding_streamer_mock, get_tagger_streamer_mock


@pytest.fixture(scope="session")
def client():
    app = create_app()

    app.dependency_overrides[
        get_danbooru_embedding_streamer
    ] = get_embedding_streamer_mock

    app.dependency_overrides[
        get_danbooru_tagger_streamer
    ] = get_tagger_streamer_mock

    with TestClient(app=app) as client:
        yield client


@pytest.fixture(scope="function")
def files():
    files = [
        (
            "images",
            (
                "temp.jpeg",
                open("src/temp.jpeg", "rb").read(),
                "image/jpeg",
            ),
        )
    ]
    return files
