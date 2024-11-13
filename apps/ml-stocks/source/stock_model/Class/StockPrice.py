from datetime import date, datetime

class StockPrice:
    
    def __init__(self, date: date, price: float, volume: int):
        
        self.date = date.date() if isinstance(date, datetime) else date
        self.price = round(price, 2)
        self.volume = volume
        
        
    def __repr__(self):
        return '{date}: {price}'.format(
                date = self.date,
                price = self.price,
            )