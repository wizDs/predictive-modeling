from datetime import date
from typing import List, Mapping
from .StockPrice import StockPrice
from .DataPair import CurrentAndFuturePair, CurrentAndPastPair

class StockProcess:
    
    def __init__(self, stockName: str, stockPrices: List[StockPrice]):
        
        self.stockName = stockName
        self.stockPrices = stockPrices
                
    
    def splitCurrPriceAndNextTDays(self, t: int) -> List[CurrentAndFuturePair]:
        
        assert len(self.stockPrices) > t, 'not enough data for t = {t} (#prices = {n})'.format(t = t, n = len(self.stockPrices))
        
        dataSplit = list()
        
        for i in range(len(self.stockPrices)):
           
            current = self.stockPrices[i]
            future  = self.stockPrices[i+1: i+1+t]
            
            if len(future) == t:
            
                dataSplit.append(CurrentAndFuturePair(current, future))
            
        return dataSplit
            

    def splitCurrPriceAndPastKDays(self, k: int) -> List[CurrentAndPastPair]:
        
        assert len(self.stockPrices) > k, 'not enough data for k = {k} (#prices = {n})'.format(k = k, n = len(self.stockPrices))
        
        dataSplit = list()
        
        for i in range(len(self.stockPrices)):
           
            current = self.stockPrices[i]
            past    = self.stockPrices[i - k: i]
            
            if len(past) == k:
            
                dataSplit.append(CurrentAndPastPair(current, past))
            
        return dataSplit
    
    def getMapper(self) -> Mapping[date, float]:
        return {s.date: s.price for s in self.stockPrices}.get
            