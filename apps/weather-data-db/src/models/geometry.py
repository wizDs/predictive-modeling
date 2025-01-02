from pydantic import BaseModel

class Geometry(BaseModel):
    coordinates: list[float]
    type: str
