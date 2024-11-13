import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from typing import List
from DataLoader import DataLoader, Multithread
from NumberOfStocksToBuy import NumberOfStocksToBuy

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


x = 10

x_years_ago   = datetime.today() - timedelta(days = 365 * x)
data_loader   = DataLoader(c25_stocks, x_years_ago, interval = 'm')

not_complete  = data_loader.data.isna().apply(any).pipe(lambda s: s[s]).index
complete_data = data_loader.data.drop(columns = not_complete)

stock_return  = complete_data.pct_change()
market_data   = complete_data.mean(axis = 1)
market_return = market_data.pct_change()

from math import isfinite

each_month = 3000

money = each_month
for r in market_return:
    if isfinite(r):
        money = (1+r) * money + each_month
        print(money)
    else:
        print(money)



x = 1

x_years_ago = datetime.today() - timedelta(days = 365 * x)
data_loader  = DataLoader(c25_stocks, x_years_ago, interval = 'm')


# 1800: 367358