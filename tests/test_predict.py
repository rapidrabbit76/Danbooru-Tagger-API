import pytest
from fastapi.testclient import TestClient


def test_predict_embedding(client: TestClient, files):
    res = client.post(
        "/predict/embedding",
        files=files,
        headers={},
    )

    assert res.ok

    embedding = res.json()

    assert isinstance(embedding[0], float)
    assert len(embedding) == 512


def test_predict_score(client: TestClient, files):
    res = client.post(
        "/predict/score",
        files=files,
        headers={},
    )

    assert res.ok

    score = res.json()

    assert isinstance(score[0], float)
    assert len(score) == 6000


def test_predict_tag(client: TestClient, files):
    res = client.post(
        "/predict/tag",
        files=files,
        json={"threshold": 0.2},
        headers={},
    )

    assert res.ok

    tags = res.json()
    assert isinstance(tags[0], str)
    assert len(tags) <= 6000
