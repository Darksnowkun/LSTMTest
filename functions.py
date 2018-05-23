from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

def F(n):
    ii += 1
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)

