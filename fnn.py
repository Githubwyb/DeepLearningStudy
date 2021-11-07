# coding=utf-8

import numpy as np

# use sigmoid function as activate function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


