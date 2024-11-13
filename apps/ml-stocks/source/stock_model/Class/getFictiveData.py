import pandas as pd
from datetime import date, timedelta
from typing import List
from .StockProcess import StockProcess
from .TrainingData import TrainingData
from .StockPrice import StockPrice


def getFictiveData(td: TrainingData, fictivePrices: List[float]) -> TrainingData: 
        
        t = td.t
        k = td.k
        
        dates = [td.currentDate]
        
        for p in fictivePrices:
            dates.append(next_workday(dates[-1]))
        
        # remove the first date, which is trainingData.currentDate
        dates = dates[1:]
        
        prevPrices         = [StockPrice(**p.__dict__) for p in td.stockPrices[-(t * 2 + 1):]]
        fictiveStockPrices = [StockPrice(date, price, None) for date, price in zip(dates, fictivePrices)]
        
        
        newProcess = StockProcess(td.stockName, prevPrices + fictiveStockPrices)
        newTrainingData = TrainingData(newProcess, k, t, td.p)
        newTrainingDataDf = pd.concat([td.X_test[:0], newTrainingData.X_test.iloc[-len(fictiveStockPrices):]]).fillna(0)
        
        
        for d in newTrainingDataDf.index:
            month = d.month
            
            if month != 1:
                newTrainingDataDf.loc[d, f"m{month}"] = 1
        
        
        return newTrainingDataDf

def next_workday(current_date: date) -> date:
    
    nextDay = current_date + timedelta(1)
    isWeekend = nextDay.isoweekday() in (6, 7)
    
    if isWeekend:
        return next_workday(nextDay)
    else:
        return nextDay