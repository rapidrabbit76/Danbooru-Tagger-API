import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("count", [i for i in range(1, 3)])
def test_predict_embedding(client: TestClient, files, count):
    res = client.post(
        "/predict/embedding",
        files=files * count,
        headers={},
    )

    assert res.ok

    embedding = res.json()

    assert isinstance(embedding[0], list)
    assert isinstance(embedding[0][0], float)
    assert len(embedding) == count
    assert len(embedding[0]) == 512


@pytest.mark.parametrize("count", [i for i in range(1, 3)])
def test_predict_score(client: TestClient, files, count):
    res = client.post(
        "/predict/score",
        files=files * count,
        headers={},
    )

    assert res.ok

    score = res.json()

    assert isinstance(score[0], list)
    assert isinstance(score[0][0], float)
    assert len(score) == count
    assert len(score[0]) == 6000


@pytest.mark.parametrize("count", [i for i in range(1, 3)])
def test_predict_tag(client: TestClient, files, count):
    res = client.post(
        "/predict/tag",
        files=files * count,
        data={"threshold": 0.2},
        headers={},
    )

    assert res.ok

    tags = res.json()
    assert isinstance(tags[0], list)
    assert isinstance(tags[0][0], str)
    assert len(tags) == count
    assert len(tags[0]) <= 6000
