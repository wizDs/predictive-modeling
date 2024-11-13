import json
from datetime import date

class ModelScore:
    
    def __init__(self, date: date, score: float):
        self.date  = date.__str__()
        self.score = score
        

    def toJson(self, path):
        
        with open(path, 'w') as outfile:
            json.dump(self.__dict__, outfile)
            
    def __repr__(self):
        
        return '{date}: {score}'.format(**self.__dict__)