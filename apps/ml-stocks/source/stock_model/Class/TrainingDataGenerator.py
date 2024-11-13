import pandas_datareader.data as web
import pandas as pd
from datetime import date
from typing import Optional
from .PriceType import PriceType
from .StockPrice import StockPrice
from .StockProcess import StockProcess
from .TrainingData import TrainingData

class TrainingDataGenerator:
    
    def __init__(self, k: int = 30, t: int = 30, p: float = 0.02):
        self.k = k
        self.t = t
        self.p = p


    def byStockName(self, name: str, start: Optional[date] = None, priceType: PriceType = PriceType.Open) -> TrainingData:
                
        data = web.get_data_yahoo(name, start = start)\
                    .reset_index()\
                    .rename(str.lower, axis = "columns")\
                    .loc[:,['date', priceType.value, 'volume']]
        
        stockPrices = [StockPrice(*s) for s in data.values]
        stockProcess = StockProcess(name, stockPrices)
        trainingData = TrainingData(stockProcess, self.k, self.t, self.p)
        
        return trainingData


    def byCsvFile(self, path: str, **kwargs) -> TrainingData:
        
        data = pd.read_csv(path, **kwargs)
        name = path.split("/")[-1].replace(".csv", "")
        
        stockPrices = [StockPrice(*s) for s in data.values]
        stockProcess = StockProcess(name, stockPrices)
        trainingData = TrainingData(stockProcess, self.k, self.t, self.p)
        
        return trainingData
        
        
        