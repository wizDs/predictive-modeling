from fastapi import FastAPI, Query
from datetime import date
from copy import deepcopy
from stock_model import TrainingDataGenerator
from stock_model import RandomForrestModel
from stock_model import LinearModel
from stock_model import EvaluationData
from purchase_tools import n_stocks_to_buy
from typing import List, Optional

app = FastAPI()

    
@app.get("/{stockName}")
async def buy_score(stockName: str, k: int = 30, t: int = 30, p: float = 0.02, n_estimators: int = 100):
    
    tdGenerator = TrainingDataGenerator(k, t, p)
    td          = tdGenerator.byStockName(stockName, start = date(2011, 1, 1)) 
    fullModel   = RandomForrestModel(td, n_estimators = n_estimators)
    summary     = fullModel.summary(td.stockPriceMapper).iloc[-1]
    
    return {summary.date.__str__(): summary.drop('date').to_dict()}


@app.get("/rf/{stockName}")
async def rf_eval(stockName: str, k: int = 30, t: int = 30, p: float = 0.02, n_estimators: int = 100, days: int = 1):
    
    assert 1 <= days <= t, "days must be between 1 and t"
    
    tdGenerator = TrainingDataGenerator(k, t, p)
    td          = tdGenerator.byStockName(stockName, start = date(2011, 1, 1))
    
    scores      = dict()
    
    for i in range(days):
        
        if i == 0:
            evalData        = deepcopy(td)
            
        else:
            
            X_train = td.X_train.iloc[:-i].copy()
            y_train = td.y_train.iloc[:-i].copy()
            X_test  = td.unlabeledData.iloc[-i - 1].copy()
            y_test  = None
            
            evalData  = EvaluationData(X_train, y_train, X_test, y_test, t)
            
        
        fullModel = RandomForrestModel(evalData, n_estimators = n_estimators) 
        summary   = fullModel.summary(td.stockPriceMapper)
        summary   = summary.round({'score': 2})
        summary = summary.sort_values('date', ascending = False).iloc[0]
        scores[summary.date.__str__()] = dict(summary.drop('date'))
    
    return scores

@app.get("/lr/{stockName}")
async def lr_eval(stockName: str, k: int = 30, t: int = 30, p: float = 0.02, days: int = 1):
    
    assert 1 <= days <= t, "days must be between 1 and t"
    
    tdGenerator = TrainingDataGenerator(k, t, p)
    td          = tdGenerator.byStockName(stockName, start = date(2011, 1, 1))
    
    scores      = dict()
    
    for i in range(days):
        
        if i == 0:
            evalData        = deepcopy(td)
            
        else:
            
            X_train = td.X_train.iloc[:-i].copy()
            y_train = td.y_train.iloc[:-i].copy()
            X_test  = td.unlabeledData.iloc[-i - 1].copy()
            y_test  = None
            
            evalData  = EvaluationData(X_train, y_train, X_test, y_test, t)
            
        
        fullModel = LinearModel(evalData) 
        summary   = fullModel.summary(td.stockPriceMapper)
        summary   = summary.round({'score': 2})
        summary   = summary.sort_values('date', ascending = False).iloc[0]
        scores[summary.date.__str__()] = dict(summary.drop('date'))
    
    return scores

@app.get("/n_buy/")
async def numbers_of_stocks_to_buy(m: int, p: Optional[List[float]] = Query(None), w: Optional[List[float]] = Query(None)):
    
    return n_stocks_to_buy(prices = p, weights = w, money = m)
    