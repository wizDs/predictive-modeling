import numpy as np
from typing import List
from math import floor
from scipy.optimize import linprog
from gekko import GEKKO


def n_stocks_to_buy(prices: List[float], weights: List[float], money: int, integer_problem: bool = True):
    
    assert len(weights) == len(prices), 'weights and prices must be of same dimension'
    assert sum(weights) <= 1,           'all weights must sum to less or equal to 1'
    
    d = len(weights)
    
    c = np.array(prices)
    b = np.array(weights) * money
    A = np.zeros((d, d), int)
    np.fill_diagonal(A, prices) 

    if integer_problem:
        res = mixed_integer_problem(c, A, b)
    else:
        res = linear_programming(c, A, b)
    

    n_buy       = [int(x) for x in res]
    total_price = round(np.dot(np.array(prices),  n_buy))

    output = {
        'total_price':  total_price,
        'prices':       prices,
        'n_buy':        n_buy,
        'slack' :       money - total_price,
    }

    return output
    
    
def mixed_integer_problem(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> List[float]:
    
    m = GEKKO() # Initialize gekko
    m.options.SOLVER=1  # APOPT is an MINLP solver
    
    d = len(c)
    
    x = np.array([m.Var(value=0, integer=True) for i in range(d)])
    
    m.Obj(-np.dot(c, x))
    
    for i in range(d):
        m.Equation([np.dot(A[i], x) <= b[i]])
    
    m.solve(disp=False) # Solve
    
    return [v.value[0] for v in x]

def linear_programming(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> List[float]:
    res = linprog(-c, A_eq=A, b_eq=b, bounds = [(0, None) for i in range(len(c))])
    
    return [floor(v) for v in res.x]
    
                 