import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date

from typing import List, Optional, Mapping
#from DataLoader import DataLoader, Multithread
import pandas_datareader.data as web
#from NumberOfStocksToBuy import NumberOfStocksToBuy

from Class.TrainingDataGenerator import TrainingDataGenerator
from Class.TrainingData import TrainingData
from Class.Regression import RandomForrestModel
from Class.ModelScore import ModelScore
from Class.PriceType import PriceType

c25_stocks = [
    'FLS.CO',
    'ISS.CO',
    'TRYG.CO',
    'SIM.CO',
    'RBREW.CO',
    'DEMANT.CO',
    'AMBU-B.CO',
    'NETC.CO',
    'NZYM-B.CO',
    'CHR.CO',
    'NOVO-B.CO',
    'LUN.CO',
    'BAVA.CO',
    'CARL-B.CO',
    'DANSKE.CO',
    'COLO-B.CO',
    'MAERSK-B.CO',
    'MAERSK-A.CO',
    'DSV.CO',
    'VWS.CO',
    'GN.CO',
    'GMAB.CO',
    'ORSTED.CO',
    'ROCK-B.CO',
    'PNDORA.CO',
]


stockName = 'GN.CO'
# stockName = 'TRYG.CO'
# stockName = 'COLO-B.CO'
# stockName = 'NOVO-B.CO'
# stockName = 'SPIC25KL.CO'


tdGenerator = TrainingDataGenerator()

import pandas_datareader.data as web
import pandas as pd
from datetime import date
from enum import Enum
from typing import Optional
from Class.StockPrice import StockPrice
from Class.StockProcess import StockProcess
from Class.TrainingData import TrainingData
from Class.getFictiveData import getFictiveData


# data = web.get_data_yahoo(stockName, start = date(2017, 1, 1))\
#             .reset_index()\
#             .rename(str.lower, axis = "columns")\
#             .loc[:,['date', PriceType.Open.value, 'volume']]

td        = tdGenerator.byStockName('CHR.CO', start = date(2011, 1, 1)) 
fullModel = RandomForrestModel(td)
summary   = fullModel.summary(td.stockPriceMapper)


    
fictive =getFictiveData(td, [200, 350])

[fullModel.predictSeries(s) for i, s in fictive.iterrows()]



    
rfPred = RandomForrestModel(trainingData)
rfPred.summary(trainingData.stockPriceMapper)



summaries = dict()

for stockName in c25_stocks:
    
    td        = tdGenerator.byStockName(stockName, start = date(2011, 1, 1)) 
    fullModel = RandomForrestModel(td)
    summary   = fullModel.summary(td.stockPriceMapper)
    
    summaries[stockName] = summary

    td.plotTimeSeries()


evaluationDataList = [td.splitDataAtIndex(index = i) for i in range(1000, len(td.X) + 1)]

ed=evaluationDataList[1000]

fullModel = RandomForrestModel(td)
pred = [fullModel.predictSeries(series) for series in td.X_test.iterrows()]
pred = fullModel.predict(td.X_test)
list(map(fullModel.scoreMapper, pred))


import json
ModelScore(ed.currentDate, pred).toJson(f'{ed.currentDate}.txt')






listPredictions = list()

for ed in evaluationDataList:
    
    rf = ModelPrediction(ed)    
    rf.toPickle(f'Models/{ed.currentDate}.pickle')

    pred = rf.predictSeries(ed.X_test)
    listPredictions.append(pred)
    


z=pd.DataFrame(listPredictions)
z['Price']= z.index.map(td.stockPriceMapper)
z['RegressionForrest'] = z.eval('sqrt(RegressionForrest / 30) * 10')

#z.to_csv('')

a = z.copy()
a = a[pd.Series(a.index).between(date(2016, 1, 1), date(2016, 5, 1)).values]








import matplotlib.pyplot as plt
import seaborn as sns

# create figure and axis objects with subplots()
fig, ax = plt.subplots(dpi = 150)
# make a plot
sns.lineplot(a.date, a.price, color="red")
# set x-axis label
ax.set_xlabel("",fontsize=14)
# set y-axis label
ax.set_ylabel("Stock Price (GN Store)",color="red",fontsize=14)

ax.set_title("High model score means high recommendation for buying")
plt.xticks(rotation=90)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
sns.lineplot(a.date, a.score,color="blue")
ax2.set_ylabel("Model score",color="blue",fontsize=14)

plt.show()

# save the plot as a file
fig.savefig('model_performance.jpg',
            format='jpeg',
            dpi=200,
            bbox_inches='tight')

