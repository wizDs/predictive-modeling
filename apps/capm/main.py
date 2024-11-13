import numpy as np
import re
import pandas as pd
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=2)


def clean_name(string: str) -> str:
    '''remove invalid characters'''
    string = re.sub(pattern='[-:().&]', repl=' ', string=string)
    string = re.sub(pattern="'", repl=' ', string=string)
    string = re.sub(pattern=':', repl=' ', string=string)
    string = re.sub(pattern='\\s+', repl='_', string=string)
    string = string.strip('_')
    string = string.lower()
    return string


df = (pd.read_excel('stoxx600monthly.xlsx')
      .set_index('Name')
      .rename(columns=clean_name))

name2code = df.loc['Code'].to_dict()
df.drop(index='Code', inplace=True)
df.reset_index().astype({'Name': 'date'})

market_value = df.filter(like='_market_value')
price = df[[name
            for name in df.columns
            if name not in set(market_value.columns)]]
return_rate = price.pct_change(periods=1)

# market monthly return rate
market_return_rate = return_rate.mean(axis=1).rename('market_return_rate')

pd.concat([return_rate, market_return_rate], axis=1)

stock_index = 6
stock_name = return_rate.columns[stock_index]
data = pd.concat([return_rate[stock_name],
                  market_return_rate], axis=1).dropna()

beta, alpha = np.polyfit(data["market_return_rate"], data[stock_name], deg=1)
print(stock_name, beta, alpha)

pp.pprint(market_value.columns.tolist())
pp.pprint(price.columns.tolist())
# print(df.columns.tolist())
