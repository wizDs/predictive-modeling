from datetime import date
from typing import List
from .Measurement import Measurement

class Features:
    
    def __init__(self, currentDate: date, pastMeasurements: List[Measurement]):
        self.date   = currentDate
        self.values = [x.value for x in pastMeasurements]
        
    def prepareForPd(self):
        
        featuresWithCurrentDate = {'date': self.date}
        for i, featureValue in enumerate(self.values):
            featuresWithCurrentDate[i] = featureValue
            
        return featuresWithCurrentDate
        