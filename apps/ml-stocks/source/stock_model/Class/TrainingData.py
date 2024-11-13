import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder

from .StockProcess import StockProcess
from .DataPair import DataPair
from .Measurement import Measurement
from .Features import Features

class EvaluationData:
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, t: int):
        
        
        self.X_train     = X_train
        self.y_train     = y_train
        self.X_test      = X_test
        self.y_test      = y_test
        
        self.t           = t
        self.currentDate = self.X_test.name if isinstance(self.X_test, pd.Series) else self.X_test.iloc[-1].name
        
    def __repr__(self):
        
        emptySet = len(self.X_train) > 0
        start    = min(self.X_train.index) if emptySet else ''
        end      = max(self.X_train.index) if emptySet else ''
        
        
        return 'EvaluationData(current date: {currentDate}, training period: {period})'.format(
                    currentDate   = self.currentDate,
                    period         = f'{start} - {end}' if emptySet else 'None',
                )


class TrainingData(EvaluationData):
    
    def __init__(self, stockProcess: StockProcess, k: int, t: int, p: float):
        
        self.t = t
        self.k = k
        self.p = p
        self.stockName       = stockProcess.stockName
        self.stockPrices     = stockProcess.stockPrices
        self.stockPriceMapper= stockProcess.getMapper()
        
        # Future (labels)
        self.currAndFuture   = stockProcess.splitCurrPriceAndNextTDays(t = t)        
        self.nFutureDaysMoreExpensiveThanCurrentPrice = self.countDaysMoreExpensiveThanCurrent(self.currAndFuture, p = p)
        self.futurePrices    = self.comparedPricesPerCurrentDate(self.currAndFuture)
        
        # Past (features)
        self.currAndPast     = stockProcess.splitCurrPriceAndPastKDays(k = k)
        self.nPastDaysMoreExpensiveThanCurrentPrice   = self.countDaysMoreExpensiveThanCurrent(self.currAndPast, p = p)
        self.pastPrices      = self.comparedPricesPerCurrentDate(self.currAndPast)
        self.featuresForCurrentPrice = self.splitCurrentEntityAndPast(self.nPastDaysMoreExpensiveThanCurrentPrice, k = k)
        
        
        # For each date, join the past (features) and the future (label)
        dataframe          = self.joinFeaturesAndLabels()
        
        # local trends (features)
        returnPerDayFromButtomMapper = {m.date : m.value for m in self.returnPerDayFromButtom(self.currAndPast)}
        returnPerDayFromTopMapper    = {m.date : m.value for m in self.returnPerDayFromTop(self.currAndPast)}
        
        dataframe['return_buttom'] = dataframe.index.map(returnPerDayFromButtomMapper)
        dataframe['return_top']    = dataframe.index.map(returnPerDayFromTopMapper)
        
        
        # Add trend feature
        dataframe['trend'] = self.trendFeature(dataframe.index)
        
        # Merge seasonality component into training data frame
        seasonality        = self.seasonalityDummies(dataframe.index).rename(mapper=lambda m: 'm{}'.format(m), axis = 1)
        dataframe = dataframe.merge(seasonality, left_index = True, right_index = True, how = 'left')
        
       
        
        # split labeled and unlabeled data
        self.labeledData   = dataframe.query("label.notna()")
        self.unlabeledData = dataframe.query("label.isna()").drop(columns = ['label'])
        
        # split into features (X) and labels (y)
        X = self.labeledData.drop(columns = ['label']).copy()
        y = self.labeledData.label.copy()
        
        EvaluationData.__init__(self, X, y, self.unlabeledData, y_test = None, t = t)
    
    def countDaysMoreExpensiveThanCurrent(self, dataPairList: List[DataPair], p: float) -> List[Measurement]:
                
        measurements = list()
        
        for dataPair in dataPairList:
            
            measurement = sum(1 for x in dataPair.compared if dataPair.current.price * (1 + p) < x.price)
            measurements.append(Measurement(dataPair.current.date, measurement))
            
        return measurements

    def returnPerDayFromButtom(self, dataPairList: List[DataPair]) -> List[Measurement]:
                    
            measurements = list()
            
            for dataPair in dataPairList:
                
                pastPrices     = sorted(dataPair.compared, key = lambda p: p.date, reverse=True)
                cheapest       = min(pastPrices, key = lambda p: p.price)
                nDaysFromToday = [i + 1 for i, x in enumerate(pastPrices) if x.date == cheapest.date][0]
                
                stockReturn = (dataPair.current.price / cheapest.price) / cheapest.price * 100
                measurement = stockReturn / nDaysFromToday
                measurements.append(Measurement(dataPair.current.date, measurement))
                
            return measurements            
    
    def returnPerDayFromTop(self, dataPairList: List[DataPair]) -> List[Measurement]:
                    
            measurements = list()
            
            for dataPair in dataPairList:
                
                pastPrices     = sorted(dataPair.compared, key = lambda p: p.date, reverse=True)
                mostExpensive  = max(pastPrices, key = lambda p: p.price)
                nDaysFromToday = [i + 1 for i, x in enumerate(pastPrices) if x.date == mostExpensive.date][0]
                
                stockReturn = (dataPair.current.price - mostExpensive.price) / mostExpensive.price * 100
                measurement = stockReturn / nDaysFromToday
                measurements.append(Measurement(dataPair.current.date, measurement))
                
            return measurements            
            
        
    def splitCurrentEntityAndPast(self, measurements: List[Measurement], k: int) -> List[Features]:
        
        listOfFeatures = list()
        
        for i in range(len(measurements)):
                       
            if i >= k:
                
                currentDate      = measurements[i].date
                pastMeasurements = measurements[i - k + 1: i + 1]
                
                listOfFeatures.append(Features(currentDate, pastMeasurements))
                    
            
        return listOfFeatures
        
        
    def comparedPricesPerCurrentDate(self, dataPairList: List[DataPair]):
    
        comparedPrices = dict()
        
        for dataPair in dataPairList:
            
            currentDate = dataPair.current.date
            comparedPrices[currentDate] = [x.price for x in dataPair.compared]

        return comparedPrices
    
    def joinFeaturesAndLabels(self):
        
        labelsDf   = pd.DataFrame([m.prepareForPd() for m in self.nFutureDaysMoreExpensiveThanCurrentPrice])
        featuresDf = pd.DataFrame([f.prepareForPd() for f in self.featuresForCurrentPrice])
        
        finalDf = featuresDf.merge(labelsDf, on = 'date', how = 'left').set_index('date')
        
        
        return finalDf
    
    def pctPriceChange(self, dataPairList: List[DataPair]):
        
        features = list()
        
        for dataPair in dataPairList:
            
            pastAndCurrentPrices = [x.price for x in dataPair.compared] + [dataPair.current.price]
            measurementValues = pd.Series(pastAndCurrentPrices).pct_change()
            measurements      = [Measurement(d, m) for d, m in zip([x.date for x in dataPair.compared], measurementValues)]
                        
            features.append(Features(dataPair.current.date, measurements))
            
        return features
            
    def seasonalityDummies(self, dates: List[date]) -> pd.DataFrame:
        
        months = pd.DataFrame({'month': [d.month for d in dates]})
        ohe    = OneHotEncoder(drop = 'first')
        season = ohe.fit_transform(months).toarray()
        
        return pd.DataFrame(season, index = dates, columns = np.sort(months.month.unique())[1:]).astype(int)
    
    def trendFeature(self, dates: List[date]) -> List[int]:
        return [d.year - 2000 for d in dates]


    def plotTimeSeries(self, title: Optional[str] = None):
        
        dates  = self.labeledData.index.copy()
        prices = dates.map(self.stockPriceMapper)
        
        # rename the column with symbole name
        price = pd.DataFrame({self.stockName: prices}, index = dates)
        ax = price.plot(title = title)
        ax.set_xlabel('date')
        ax.set_ylabel('closing price')
        ax.grid()
        plt.show()
    
    def splitDataAtIndex(self, index: int) -> EvaluationData:
        
        data = self.labeledData.copy()
        
        availableDataAtTimeX         = data.iloc[:index+1]        
        availableLabeledData = availableDataAtTimeX.iloc[:-(self.t)]
        currentData                  = availableDataAtTimeX.iloc[-1]
        
        
        X_train     = availableLabeledData.drop(columns =['label'])
        y_train     = availableLabeledData.label
        X_test      = currentData.drop(index = ['label'])
        y_test      = currentData.label
        
        
        return EvaluationData(X_train, y_train, X_test, y_test, self.t)
        
    def __repr__(self):
        
        emptySet = len(self.X_train) > 0
        start    = min(self.X_train.index) if emptySet else ''
        end      = max(self.X_train.index) if emptySet else ''
        
        
        return 'TrainingData(current date: {currentDate}, training period: {period})'.format(
                    currentDate   = self.currentDate,
                    period         = f'{start} - {end}' if emptySet else 'None',
                )
