# students_LSTM.py

import pandas as pd 
import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
from tensorflow import keras

def normalize_minmax(x, x_min, x_max, min_r, max_r):
    """ Normalizing and scaling given data
    in an array x to the range of min_r 
    to max_r. x_max and x_min are maximum
    and minium of the total data x and do
    not need to be in x itself. 
    """
    x_s = ((i - x_min)/(x_max - x_min) for i in x)
    return [i * (max_r - min_r) + min_r for i in x_s]

def de_normalize_minmax(x_scale, x, x_min, x_max, min_r, max_r):
    """ Transforms a min-max normalized and 
    scaled vector back to its un-normalized
    form.
    """
    x_t = (((i - min_r)/(max_r - min_r)) for i in x_scale)
    return [(i*(x_max - x_min) + x_min) for i in x_t]

def split_data_train_test(data_array, ratio):
    """ Splits a given array into to sperate
    arrays (training and test data). The ratio
    determines the percentage of the split. 

    Example: ratio = 0.67
    The training array includes the first 67% of the given array.
    The test array includes the next 33% of the given array.
    """
    train_size = int(len(data_array) * ratio)
    train_data = data_array[0:train_size]
    test_data = data_array[train_size:len(data_array)]

    return np.array(train_data), np.array(test_data)

# convert an array of values into a data_set matrix
def create_data_set(_data_set, _look_back=1): 
    data_x, data_y = [], []
    
    for i in range(len(_data_set) - _look_back - 1): 
        a = _data_set[i:(i + _look_back), 0] 
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0]) 
        
    return np.array(data_x), np.array(data_y)

# load the data_set 
data_frame = pd.read_csv('students_small.txt', sep=" ", header=None)
data_frame.columns = ["year", "students"]
data_set = data_frame.values 
data_set = data_set.astype('float32')

# split into train and test sets
train_size = int(len(data_set) * 0.67)
test_size = len(data_set) - train_size
train, test = data_set[0:train_size, :], data_set[train_size:len(data_set), :]

# reshape into X=t and Y=t+1
look_back = 1
train_x, train_y = create_data_set(train, look_back)
test_x, test_y = create_data_set(test, look_back)

print(train_x)
print(train_y)
print(test_x)
print(test_y)

train_y_norm = normalize_minmax(all_students_vec, 0, 500000, 0, 1)