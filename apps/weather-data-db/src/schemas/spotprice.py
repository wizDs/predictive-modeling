from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SpotPrice(BaseModel):
    HourUTC: datetime
    HourDK: datetime
    PriceArea: Optional[str] = Field(max_length=10)
    SpotPriceDKK: Optional[float]
    SpotPriceEUR: Optional[float]