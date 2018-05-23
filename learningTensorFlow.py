import numpy as np
import tensorflow as tf
import copy as cp
from functions import *

#hyperparameters
numTestCases = 200
tlen = 4
numHiddenNeurons = 50

if numTestCases/4 != int(numTestCases/4):
    raise Exception("I don't know python non dividisble seoinsoe testlength")

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
        fibLookUp.append(F(n-1)+F(n-2))
        return fibLookUp[n]

F(numTestCases+2)
for i in range(2, numTestCases, tlen):
    x.append(fibLookUp[i: i+tlen])

# ------------------ TensorFlow --------------#

data = tf.placeholder(tf.int32, [None, 4, 1])
target = tf.placeholder(tf.int32, [None, 1])
cell = tf.nn.rnn_cell.LSTMCell(numHiddenNeurons, state_is_tuple=True)
RNNOut, RNNState = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)



dehbug = 1

