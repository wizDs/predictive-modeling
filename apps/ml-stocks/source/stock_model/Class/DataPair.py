import numpy as np
from typing import List
from .StockPrice import StockPrice

class DataPair:
    
    def __init__(self, current: StockPrice, compared: List[StockPrice]):
                
        self.current = current 
        self.compared = [p for p in compared]
       


class CurrentAndFuturePair(DataPair):
    
    def __init__(self, current: StockPrice, future: List[StockPrice]):
        
        DataPair.__init__(self, current, future)
        

    def __repr__(self):
        return 'current; {curr} - future (t = {t}); {future}'.format(
                curr   = self.current,
                future = np.mean([x.price for x in self.compared]).round(2),
                t      = len(self.compared),
            )

class CurrentAndPastPair(DataPair):
    
    def __init__(self, current: StockPrice, pastPrices: List[StockPrice]):
               
        DataPair.__init__(self, current, pastPrices)
        

    def __repr__(self):
        return 'current; {curr} - past (k = {k}); {past}'.format(
                curr   = self.current,
                past = np.mean([x.price for x in self.compared]).round(2),
                k      = len(self.compared),
            )

    