__author__ = 'Haohan Wang'

import numpy as np

def centralize(x):
    m = np.mean(x)
    return x - m

def mapping2ZeroOne(x):
    maxi = np.max(x)
    mini = np.min(x)
    return (x-mini)/(maxi-mini)

def rescale(x):
    maxi = np.max(np.abs(x))
    if maxi == 0:
        return x
    return x/maxi

def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / s