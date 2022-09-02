from pydantic import BaseModel


class Tag(BaseModel):
    name: str
    score: float
