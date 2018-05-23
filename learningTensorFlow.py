import numpy as np
import tensorflow as tf
import copy as cp
from functions import *

#hyperparameters
numTestCases = 200
tlen = 4
ii = 0

#inits

#x = np.zeros((numTestCases, tlen))
x = []

#Fibb Input Data
#for i in range(0, numTestCases):
#    x[i, 0] = i


#Fibb Output Data
fibLookUp = [0, 1, 1]
def F(n):
    if len(fibLookUp) > n:
        return fibLookUp[n]
    else: 
        fibLookUp[n] = F(n-1)+F(n-2)
        return fibLookUp[n]

F(numTestCases+2)
for i in range(2, numTestCases, tlen):
    x.append(fibLookUp[i: i+tlen])

dehbug = 1

