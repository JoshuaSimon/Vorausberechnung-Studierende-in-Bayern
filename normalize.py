# normalize.py

import numpy as np 
from math import sqrt
from sklearn.preprocessing import minmax_scale

def normalize_l2 (x):
    """ Rerturns a normalized copy of
    an array x. Uses L2 vector norm 
    for normalization. 
    """
    l2 = sqrt(sum(i**2 for i in x))
    return [i/l2 for i in x]

def de_normalize_l2 (x_norm, x):
    """ Transforms a normalized vector
    back to its un-normalized form. 
    """
    l2 = sqrt(sum(i**2 for i in x))
    return [i*l2 for i in x_norm]

def max_scaling(x):
    """ Scaling the values of an array x
    with respect to the maximum value of x.
    """
    return [i/max(x) for i in x]

def normalize_minmax(x, min_r, max_r):
    """ Normalizing and scaling given data
    in an array x to the range of min_r 
    to max_r.
    """
    x_s = [(i - min(x))/(max(x) - min(x)) for i in x]
    return [i * (max_r - min_r) + min_r for i in x_s]

def de_normalize_minmax(x_scale, x, min_r, max_r):
    """ Transforms a min-max normalized and 
    scaled vector back to its un-normalized
    form.
    """
    x_t = [((i - min_r)/(max_r - min_r)) for i in x_scale]
    x_inv = [(i*(max(x) - min(x)) + min(x)) for i in x_t]
    return x_inv

# Test data
x = [2000, 2001, 2003]

# Exampels:
# L2 normalization
print(normalize_l2(x))
print(de_normalize_l2(normalize_l2(x), x))

# Max scaling
print(max_scaling(x))

# Min-Max scaling
print(normalize_minmax(x, 0, 10))
print(de_normalize_minmax(normalize_minmax(x, 0, 10), x, 0, 10))

# sklearn data scaling
print(minmax_scale(x, feature_range=(0, 1), axis=0, copy=False))
