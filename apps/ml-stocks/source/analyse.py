import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from stock_model import PriceType


def months_between_two_dates(start: date, end: date) -> int:
    # https://www.kite.com/python/answers/how-to-get-the-number-of-months-between-two-dates-in-python
    return (end.year - start.year) * 12 + (end.month - start.month)

class MoneyStream(object):
    
    def __init__(self, saving: int, start: date, end: date = date.today()):
        
        self.saving = saving
        self.start  = start
        self.end    = end
        
        self.months = months_between_two_dates(start, end)
        self.years  = self.months // 12
        
        
        self.stream = dict()
        self.cum_stream = dict()
        
        for m in range(self.months + 1):
        
            date_for_saving             = str(start + relativedelta(months=+m))
            self.stream[date_for_saving]     = saving 
            self.cum_stream[date_for_saving] = saving * (m + 1)
      
        
def match_curr_price(d:date, price_dict: dict, score_dict: dict = {}):
    
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime().date()
        
    price = price_dict.get(str(d))
    score = score_dict.get(str(d))
    
    return {'purchase_day': d, 'price': price, 'score': score}
        
       
def match_next_opening_price(d: date, price_dict: dict, score_dict: dict = {}, score_threshold: float = 7):
    
        
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime().date()
        
    latest_observed_price = max(d.date() for d in pd.to_datetime(list(price_map.keys())))
    potential_next_working_day = try_find_next_working_day(d)
    price = price_dict.get(str(potential_next_working_day))
    score = score_dict.get(str(potential_next_working_day))
    
    if (price is not None) and ((score is None) or (score >= score_threshold)):
        return {'purchase_day': potential_next_working_day, 'price': price, 'score': score}
    elif potential_next_working_day > latest_observed_price:
        return {'purchase_day': potential_next_working_day, 'price': None, 'score': None}
    else:
        return match_next_opening_price(potential_next_working_day, price_dict, score_dict, score_threshold)
    
        
def identify_selling_date(d: date, price_dict: dict, score_dict: dict = {}, score_threshold: float = 4):
    
        
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime().date()
        
    latest_observed_price = max(d.date() for d in pd.to_datetime(list(price_map.keys())))
    potential_next_working_day = try_find_next_working_day(d)
    price = price_dict.get(str(potential_next_working_day))
    score = score_dict.get(str(potential_next_working_day))
    
    if (price is not None) and ((score is None) or (score < score_threshold)):
        return {'purchase_day': potential_next_working_day, 'price': price, 'score': score}
    elif potential_next_working_day > latest_observed_price:
        return {'purchase_day': potential_next_working_day, 'price': None, 'score': None}
    else:
        return identify_selling_date(potential_next_working_day, price_dict, score_dict, score_threshold)

def try_find_next_working_day(d: date):
        
    dayofweek = d.weekday()
    
    if dayofweek == 5:
        return d + timedelta(2)
    elif dayofweek == 6:
        return d + timedelta(1)
    else:
        return d + timedelta(1)


def use_all_money(saldo: float, price: float, transaction_cost: float) -> dict:

    n_buy = (saldo - transaction_cost) // price
    cost = n_buy * price
    remaining = saldo - cost
    
    return {'n_buy': n_buy, 'cost': cost, 'remaining': remaining}

def sell_all_stocks(saldo: float, n_stocks: int, price: float, transaction_cost: float) -> dict:

    gain = (n_stocks * price) - transaction_cost
    remaining = saldo + gain
    
    return {'n_sell': n_stocks, 'gain': gain, 'remaining': remaining}    

def sell_quarter_of_stocks(saldo: float, n_stocks: int, price: float, transaction_cost: float) -> dict:

    n_sell = n_stocks // 4
    gain = (n_sell * price) - transaction_cost
    remaining = saldo + gain
    
    return {'n_sell': n_sell, 'gain': gain, 'remaining': remaining}    

# =============================================================================
# import scores
# =============================================================================
import os
import json 

stockname = 'gn.co'
path = f'c:/local_repos/{stockname}/'
files = os.listdir(path)
txt_files = filter(lambda name: '.txt' in name, files)

stock_scores = list()

for filename in txt_files:
    with open(path + filename, 'r') as json_file:
        stock_scores.append(json.load(json_file))

stock_scores_df  = pd.DataFrame(stock_scores)

price_map = dict(zip(stock_scores_df.date, stock_scores_df.price))
score_map = dict(zip(stock_scores_df.date, stock_scores_df.score))



# =============================================================================
# money stream
# =============================================================================
moneystream     = MoneyStream(saving = 2000, start = date(2015, 1, 27), end = date(2021, 1, 27))
moneystream_df  = pd.Series(moneystream.stream, name= 'saving').to_frame()
moneystream_df.index  = [d.to_pydatetime() for d in pd.to_datetime(moneystream_df.index)]


# =============================================================================
#     next available
# =============================================================================
next_available = list()

for d in moneystream_df.index:
    
    matched_price = match_curr_price(d, price_map)
    
    if matched_price['price'] is None:
        matched_price = match_next_opening_price(d, price_map)
    
    next_available.append(matched_price)
    
next_available = pd.DataFrame(next_available, index = moneystream_df.index).rename_axis(index = 'date')
next_available 



# =============================================================================
#     purchase time from model
# =============================================================================
purchase_from_model = list()

for i, d in enumerate(moneystream_df.index):
    
    
    if i == (moneystream_df.shape[0] - 1):
        matched_price = {'purchase_day': None, 'price': None, 'score': None}
        
    else:
        matched_price = match_curr_price(d, price_map, score_map)
        
        if (matched_price['price'] is None) or (matched_price['score'] < 7.5):
            matched_price = match_next_opening_price(d, price_map, score_map, score_threshold=7.5)

    
    purchase_from_model.append(matched_price)
    
purchase_from_model  = pd.DataFrame(purchase_from_model, index = moneystream_df.index).rename_axis(index = 'date').rename(columns = {'price': 'purchase_price', 'score': 'purchase_score'})
purchase_from_model  = purchase_from_model.drop_duplicates(keep='last')

# purchase['diff_days'] = purchase.assign(purchase_day = lambda df: ([d if pd.isnull(row.purchase_day) else row.purchase_day for d, row in df.iterrows()])).eval('purchase_day_model - purchase_day').dt.days
# purchase['diff_price']  = purchase.eval('open_price - price_model')


# =============================================================================
#     selling time from model
# =============================================================================
selling_from_model = list()

for i, d in enumerate(moneystream_df.index):
    
    if i == (moneystream_df.shape[0] - 1):
        matched_price = {'purchase_day': None, 'price': None, 'score': None}
        
    else:
        matched_price = match_curr_price(d, price_map, score_map)
        
        if (matched_price['price'] is None) or (matched_price['score'] > 3.5):
            matched_price = identify_selling_date(d, price_map, score_map, score_threshold=3.5)

    
    selling_from_model.append(matched_price)
    
selling_from_model  = pd.DataFrame(selling_from_model , index = moneystream_df.index).rename_axis(index = 'date').rename(columns = {'purchase_day': 'selling_day', 'price': 'selling_price', 'score': 'selling_score'})
selling_from_model  = selling_from_model.drop_duplicates(keep='last')


# =============================================================================
#     buy over time (without model)
# =============================================================================
saving = 2000

  


remaining = 0
for d, row in next_available.iterrows():
    
    saldo_primo = saving + remaining
    next_available.loc[d, 'saldo_primo'] = saldo_primo
    
    state = use_all_money(saldo = saldo_primo, price = row.price, transaction_cost = 29)
    remaining = state['remaining']
    
    next_available.loc[d, 'n_buy']     = state['n_buy']
    next_available.loc[d, 'cost']      = state['cost']
    next_available.loc[d, 'saldo_ultimo'] = state['remaining']
    

next_available



# =============================================================================
# purchase
# =============================================================================
purchase_from_model_stream = moneystream_df.merge(purchase_from_model, how = 'left', left_index=True, right_index=True).rename_axis(index = 'date')
purchase_from_model_stream.index = [d.to_pydatetime().date() for d in purchase_from_model_stream.index]

saving = 2000
remaining = 0
n_stocks = 0

for d, row in purchase_from_model_stream.iterrows():
        
    saldo_primo = saving + remaining
    purchase_from_model_stream.loc[d, 'saldo_primo'] = saldo_primo
    
    state = use_all_money(saldo = saldo_primo, price = row.purchase_price, transaction_cost = 29)
    # months_waited = months_between_two_dates(d, row.purchase_day)
    
    if isinstance(row.purchase_day, date):
        
        remaining = state['remaining']
        n_stocks  = n_stocks + state['n_buy']
        
        purchase_from_model_stream.loc[d, 'n_buy']        = state['n_buy']
        purchase_from_model_stream.loc[d, 'cost']         = state['cost']
        purchase_from_model_stream.loc[d, 'saldo_ultimo'] = remaining
        purchase_from_model_stream.loc[d, 'n_stocks']     = n_stocks
        
    else:
        
        remaining = remaining + saving
        
        purchase_from_model_stream.loc[d, 'n_buy'] = 0
        purchase_from_model_stream.loc[d, 'cost']  = 0
        purchase_from_model_stream.loc[d, 'saldo_ultimo'] = remaining
        purchase_from_model_stream.loc[d, 'n_stocks'] = n_stocks
        

purchase_from_model_stream

purchase_from_model_stream.merge(next_available[['price']].rename(columns = {'price' : 'price_first'}), left_index=True, right_index=True, how = 'left')

a= purchase_from_model_stream.merge(selling_from_model, how = 'left', left_index = True, right_index = True)
a['diff'] = a.eval('selling_day - purchase_day').dt.days

n_model, n= purchase_from_model_stream.n_buy.sum(), next_available.n_buy.sum()
print(n_model, n)

# =============================================================================
# purchase and sell (simple)
# =============================================================================
purchase_sell_from_model = moneystream_df.merge(purchase_from_model, how = 'left', left_index=True, right_index=True).rename_axis(index = 'date')
purchase_sell_from_model = purchase_sell_from_model.merge(selling_from_model, how = 'left', left_index = True, right_index = True)
purchase_sell_from_model.index = [d.to_pydatetime().date() for d in purchase_sell_from_model.index]

saving = 2000
remaining = 0
n_stocks = 20

for d, row in purchase_sell_from_model.iloc[:-1].iterrows():
        
    
            
    saldo_primo = saving + remaining
    purchase_sell_from_model.loc[d, 'saldo_primo'] = saldo_primo
    
    
    if isinstance(row.selling_day, date) and (n_stocks > 0):
        state = sell_all_stocks(saldo_primo, n_stocks, row.selling_price, transaction_cost = 29)
        
        # ---------------------n_stocks burde tømmes--------------------
        n_stocks  = n_stocks - state['n_sell']
        remaining = state['remaining']
        
        purchase_sell_from_model.loc[d, 'n_sell'] = state['n_sell']
        purchase_sell_from_model.loc[d, 'gain'] = state['gain']
        purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
        purchase_sell_from_model.loc[d, 'n_stocks'] = 0
                
        
    else:
        
        remaining = remaining + saving
        
        purchase_sell_from_model.loc[d, 'n_sell'] = 0
        purchase_sell_from_model.loc[d, 'gain']  = 0
        purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
        purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
    
    
purchase_sell_from_model
        
    
    
    
    
    
# =============================================================================
# purchase and sell (full)
# =============================================================================
purchase_sell_from_model = moneystream_df.merge(purchase_from_model, how = 'left', left_index=True, right_index=True).rename_axis(index = 'date')
purchase_sell_from_model = purchase_sell_from_model.merge(selling_from_model, how = 'left', left_index = True, right_index = True)
purchase_sell_from_model = purchase_sell_from_model.merge(next_available['price'].rename('first_available_price').to_frame(), how = 'left', left_index = True, right_index = True)
purchase_sell_from_model.index = [d.to_pydatetime().date() for d in purchase_sell_from_model.index]

# simplify
# purchase_sell_from_model.loc[purchase_sell_from_model.purchase_day.notna() & purchase_sell_from_model.selling_day.notna(), 'selling_day'] = None


saving = 2000
remaining = 0
n_stocks = 0

for d, row in purchase_sell_from_model.iloc[:-1].iterrows():
    
    saldo_primo = saving + remaining
    purchase_sell_from_model.loc[d, 'saldo_primo'] = saldo_primo
    
    # state_sell = sell_all_stocks(saldo_primo, n_stocks, row.selling_price, transaction_cost = 29)
    # state_sell = sell_half_of_stocks(saldo_primo, n_stocks, row.selling_price, transaction_cost = 29)
    state_sell = sell_quarter_of_stocks(saldo_primo, n_stocks, row.selling_price, transaction_cost = 29)
    
    state_buy  = use_all_money(saldo_primo, price = row.purchase_price, transaction_cost = 29)
    
    if isinstance(row.selling_day, date):
        
        if isinstance(row.purchase_day, date):
            
            if (row.selling_day < row.purchase_day):
                purchase_sell_from_model.loc[d, 'temp'] = 'sell before purchase'
                
                #--------------------------------
                if (n_stocks > 0):
                    
                    n_stocks  = n_stocks - state_sell['n_sell']
                    remaining = state_sell['remaining']
                    
                    purchase_sell_from_model.loc[d, 'n_sell'] = state_sell['n_sell']
                    purchase_sell_from_model.loc[d, 'gain'] = state_sell['gain']
                    purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
                    purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
                
                    state_buy  = use_all_money(remaining, price = row.purchase_price, transaction_cost = 29)
                else:
                    purchase_sell_from_model.loc[d, 'temp'] = 'sell before purchase (nothing to sell)'
                
                remaining = state_buy['remaining']
                n_stocks  = n_stocks + state_buy['n_buy']
                
                purchase_sell_from_model.loc[d, 'n_buy']        = state_buy['n_buy']
                purchase_sell_from_model.loc[d, 'cost']         = state_buy['cost']
                purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
                purchase_sell_from_model.loc[d, 'n_stocks']     = n_stocks
                
                
                
                
            elif (row.selling_day > row.purchase_day):
                purchase_sell_from_model.loc[d, 'temp'] = 'purchase before sell'
                
                
                remaining = state_buy['remaining']
                n_stocks  = n_stocks + state_buy['n_buy']
                
                purchase_sell_from_model.loc[d, 'n_buy']        = state_buy['n_buy']
                purchase_sell_from_model.loc[d, 'cost']         = state_buy['cost']
                purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
                purchase_sell_from_model.loc[d, 'n_stocks']     = n_stocks
                
                
                state_sell = sell_quarter_of_stocks(remaining, n_stocks, row.selling_price, transaction_cost = 29)
                
                n_stocks  = n_stocks - state_sell['n_sell']
                remaining = state_sell['remaining']
                
                purchase_sell_from_model.loc[d, 'n_sell'] = state_sell['n_sell']
                purchase_sell_from_model.loc[d, 'gain'] = state_sell['gain']
                purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
                purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
            
                
                
                
        else: 
            
            # only selling
            
            purchase_sell_from_model.loc[d, 'temp'] = 'only selling'
            
            if (n_stocks > 0):
                n_stocks  = n_stocks - state_sell['n_sell']
                remaining = state_sell['remaining']
                
                purchase_sell_from_model.loc[d, 'n_sell'] = state_sell['n_sell']
                purchase_sell_from_model.loc[d, 'gain'] = state_sell['gain']
                purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
                purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
                
            else:
                
                remaining = saldo_primo
                
                purchase_sell_from_model.loc[d, 'n_sell'] = 0
                purchase_sell_from_model.loc[d, 'gain'] = 0
                purchase_sell_from_model.loc[d, 'saldo_ultimo'] = saldo_primo
                purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
            
            
            
    else:
        if isinstance(row.purchase_day, date):
            purchase_sell_from_model.loc[d, 'temp'] = 'only buying'
            
                    
            remaining = state_buy['remaining']
            n_stocks  = n_stocks + state_buy['n_buy']
            
            purchase_sell_from_model.loc[d, 'n_buy']        = state_buy['n_buy']
            purchase_sell_from_model.loc[d, 'cost']         = state_buy['cost']
            purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
            purchase_sell_from_model.loc[d, 'n_stocks']     = n_stocks
            
            
        else:
            
            remaining = remaining + saving
            
            # buy
            purchase_sell_from_model.loc[d, 'n_buy'] = 0
            purchase_sell_from_model.loc[d, 'cost']  = 0
            purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
            purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
            
            # sell
            purchase_sell_from_model.loc[d, 'n_sell'] = 0
            purchase_sell_from_model.loc[d, 'gain']  = 0
            purchase_sell_from_model.loc[d, 'saldo_ultimo'] = remaining
            purchase_sell_from_model.loc[d, 'n_stocks'] = n_stocks
            
            
            
        #     saldo_primo = saving + remaining
        #     purchase_from_model_stream.loc[d, 'saldo_primo'] = saldo_primo
            
        #     state = use_all_money(saldo = saldo_primo, price = row.price, transaction_cost = 29)
        #     # months_waited = months_between_two_dates(d, row.purchase_day)
            
            
        #     if isinstance(row.purchase_day, date):
            
        #         remaining = state['remaining']
        #         n_stocks  = n_stocks + state['n_buy']
                
        #         purchase_from_model_stream.loc[d, 'n_buy'] = state['n_buy']
        #         purchase_from_model_stream.loc[d, 'cost'] = state['cost']
        #         purchase_from_model_stream.loc[d, 'saldo_ultimo'] = state['remaining']
        #         purchase_from_model_stream.loc[d, 'n_stocks'] = n_stocks
                
        #     else:
                
        #         remaining = remaining + saving
                
        #         purchase_from_model_stream.loc[d, 'n_buy'] = 0
        #         purchase_from_model_stream.loc[d, 'cost']  = 0
        #         purchase_from_model_stream.loc[d, 'saldo_ultimo'] = remaining
        #         purchase_from_model_stream.loc[d, 'n_stocks'] = n_stocks
        

        
print(n_model, n)
purchase_sell_from_model

purchase_from_model.merge(next_available[['price']].rename(columns = {'price' : 'price_first'}), left_index=True, right_index=True, how = 'left')

a['diff'] = a.eval('selling_day - purchase_day').dt.days

n_model, n= purchase_from_model.n_buy.sum(), next_available.n_buy.sum()
print()
purchase_from_model.iloc[-1].price
next_available.iloc[-1].price




# =============================================================================
# scoring af ting
# =============================================================================
import pandas_datareader.data as web
import pandas as pd
from datetime import date
from typing import Optional
from stock_model import PriceType
from stock_model import RandomForrestModel
from stock_model import TrainingDataGenerator
import json
stockname = 'tryg.co'

tdGenerator = TrainingDataGenerator(k= 30, t = 30, p= 0.02)
td          = tdGenerator.byStockName(stockname, start = date(2011, 1, 1)) 


n = td.labeledData.shape[0]

for i in range(200, n):
    splitted_data = td.splitDataAtIndex(i)
    
    fullModel   = RandomForrestModel(splitted_data, n_estimators = 200)
    summary     = fullModel.summary(td.stockPriceMapper).iloc[-1].to_dict()
    summary['date'] = str(summary['date'])
    
    today = summary['date']
    filename = f'c:/local_repos/{stockname}/{today}.txt'
    print(today)

    with open(filename, 'w') as json_file:
        json.dump(summary, json_file)


# =============================================================================
# plot
# =============================================================================
import matplotlib.pyplot as plt
from pandas import Timestamp

df = stock_scores_df.copy()
df['date'] = pd.to_datetime(df.date)
df['year'] = df.date.dt.year
df = df.query("year == 2013")
print(df.corr())


ax = df.plot(x="date", y="score", legend=False)
ax2 = ax.twinx()
df.plot(x="date", y="price", ax=ax2, legend=False, color="r")
ax.figure.legend()
ax.hlines(y = 7.5, xmin = df['date'].min(), xmax = df['date'].max(), color = 'grey', linestyles = 'dashed')
ax.hlines(y = 3.5, xmin = df['date'].min(), xmax = df['date'].max(), color = 'grey', linestyles = 'dashed')
plt.show()

