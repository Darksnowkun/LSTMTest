"""import numpy as np
import tensorflow as tf
import copy as cp
from functions import *

#hyperparameters
numTestCases = 200
tlen = 4
ii = 0

#inits
x = np.zeros((numTestCases, tlen))

#Fibb Input Data
for i in range(0, numTestCases):
    x[i, 0] = i
"""
#Fibb Output Data
fibLookUp = dict()
def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    elif n in fibLookUp:
        return fibLookUp[n]
    else: 
        fibLookUp[n] = F(n-1)+F(n-2)
        return fibLookUp[n]
print(F(20))




"""
x = [
    [1, 1, 2, 3],
    [2, 3, 5, 8],
    [21, 34, ...]
     ]"""