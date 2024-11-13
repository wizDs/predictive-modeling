import pandas as pd
import pickle
from datetime import date
from typing import Mapping, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from .TrainingData import EvaluationData
from .ModelScore import ModelScore


class RandomForrestModel(RandomForestRegressor):
    
    def __init__(self, ed: EvaluationData, n_estimators: int = 1000, random_state: Optional[int] = None):
        
        RandomForestRegressor.__init__(self, n_estimators = n_estimators, random_state = random_state)# Instantiate model with 1000 decision trees
        self.fit(ed.X_train, ed.y_train);    
        self.t = ed.t
    
        self.modelScores = self.predictTestData(ed)
    
    
    def predictTestData(self, ed: EvaluationData):
        
        if type(ed.X_test) == pd.Series:
            
            score      = self.predictSeries(ed.X_test)
            modelScore = [ModelScore(ed.currentDate, score)]
                
            return modelScore
        
        
        elif type(ed.X_test) == pd.DataFrame:
            
            scores      = self.predict(ed.X_test)
            scores      = map(self.scoreMapper, scores)
            modelScores = [ModelScore(date, score) for date, score in zip(ed.X_test.index, scores)]
            
            return modelScores
        
      
    def predictSeries(self, series: pd.Series, normalize: bool = True) -> float:
        
        X      = series.values.reshape(1, -1)
        y_pred = self.predict(X)[0]
                
        return self.scoreMapper(y_pred) if normalize else y_pred
    
    
    def scoreMapper(self, x: float) -> float:
        
        score = (x / self.t) ** 0.6 * 10.5
        
        return min(score, 10)
    
    def toPickle(self, path: str):

        pickle.dump(self, open(path, 'wb'))
        
        
    def readPickle(path: str):
        return pickle.load(open(path, 'rb'))
    
    def summary(self, priceMapper: Mapping[date, float] = None) -> pd.DataFrame:
        
        data = pd.DataFrame([ms.__dict__ for ms in self.modelScores])
        data['date']  = pd.to_datetime(data['date']).dt.date
        data['price'] = None if priceMapper is None else data['date'].map(priceMapper)
            
        return data
    
            

class LinearModel(LinearRegression):
  
    def __init__(self, ed: EvaluationData):
        
        LinearRegression.__init__(self)
        self.fit(ed.X_train, ed.y_train)
        self.t = ed.t
        
        self.modelScores = self.predictTestData(ed)
      
    
    def predictTestData(self, ed: EvaluationData):
        
        if type(ed.X_test) == pd.Series:
            
            score      = self.predictSeries(ed.X_test)
            modelScore = [ModelScore(ed.currentDate, score)]
                
            return modelScore
        
        
        elif type(ed.X_test) == pd.DataFrame:
            
            scores      = self.predict(ed.X_test)
            scores      = map(self.scoreMapper, scores)
            modelScores = [ModelScore(date, score) for date, score in zip(ed.X_test.index, scores)]
            
            return modelScores
        
      
        
    def predictSeries(self, series: pd.Series, normalize: bool = True) -> float:
        
        X      = series.values.reshape(1, -1)
        y_pred = self.predict(X)[0]
                
        return self.scoreMapper(y_pred) if normalize else y_pred
    
    
    def scoreMapper(self, x: float) -> float:
        
        score = (x / self.t) ** 0.6 * 10.5
        
        return min(score, 10)
    
    def toPickle(self, path: str):

        pickle.dump(self, open(path, 'wb'))
        
        
    def readPickle(path: str):
        return pickle.load(open(path, 'rb'))
    
    def summary(self, priceMapper: Mapping[date, float] = None) -> pd.DataFrame:
        
        data = pd.DataFrame([ms.__dict__ for ms in self.modelScores])
        data['date']  = pd.to_datetime(data['date']).dt.date
        data['price'] = None if priceMapper is None else data['date'].map(priceMapper)
            
        return data
    