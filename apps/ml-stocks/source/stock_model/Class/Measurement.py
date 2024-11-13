from datetime import date

class Measurement:
    '''
    A measurement is a point aggregate/ feature value, for instance
    pct return for a stock ect.
    '''
    
    def __init__(self, date: date, value: float):
        self.date   = date
        self.value  = value
        
    def prepareForPd(self):
        
        return {'date': self.date, 'label': self.value}
        