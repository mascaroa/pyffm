import numpy as np


def logistic(x):
    return np.divide(1, (1 + np.exp(-x)))
