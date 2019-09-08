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

# Test data
x = [2000, 2001, 2003]

# L2 normalization
print(normalize_l2(x))
print(de_normalize_l2(normalize_l2(x), x))

# sklearn data scaling
print(minmax_scale(x, feature_range=(0, 1), axis=0, copy=False))
