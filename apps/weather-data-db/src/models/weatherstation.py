from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from src.models.geometry import Geometry  # type: ignore[import-untyped]


class Property(BaseModel):
    barometerHeight: Optional[float]
    country: str
    created: datetime
    name: str
    operationFrom: datetime
    operationTo: Optional[str]
    owner: str
    parameterId: list[str]
    regionId: Optional[str]
    stationHeight: Optional[float]
    stationId: str
    status: str
    type: str
    validFrom: datetime
    validTo: Optional[datetime]


class WeatherStation(BaseModel):
    geometry: Geometry
    id: str
    properties: Property
