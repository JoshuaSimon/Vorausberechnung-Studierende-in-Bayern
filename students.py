# students.py

import pandas as pd 
import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def linear_regression (x_data, y_data):
    """ Linear regression via least square method. 
    Returns slope and intercept for a linear fit 
    of given x and y data. 
    """
    x_mean = sum(x_data) / np.prod(x_data.shape)
    y_mean = sum(y_data) / np.prod(y_data.shape)

    sxx = 0
    syy = 0
    sxy = 0

    for i in range(np.prod(x_data.shape)):
        sxx = sxx + (x_mean - x_data[i])**2
        syy = syy + (y_mean - y_data[i])**2
        sxy = sxy + (x_mean - x_data[i]) * (y_mean - y_data[i])

    mx = sxy / sxx
    my = sxy / syy

    bx = y_mean - mx * x_mean
    by = y_mean - my * x_mean

    return mx, my, bx, by

def linear_function(m, x, b):
    """ A linear function of one variable x 
    with slope m and and intercept b. 
    """
    return m * x + b

def lin_predict(x, x_data, y_data):
    """ Predicts a value for a given data set
    with an linear realation of X and Y.
    """
    mx, my, bx, by = linear_regression(x_data, y_data)

    x_predict = []
    y_predict = []

    for i in x:
        x_predict.append(linear_function(mx, i, bx))
        y_predict.append(linear_function(my, i, by))
    
    return x_predict, y_predict

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


# Read the data form the input file and name the columns. 
student_data = pd.read_csv("student_data.txt", sep=" ", header=None)
student_data.columns = ["semester", "year", "all", "all_male", "all_female", 
                        "all_ger", "male_ger", "female_ger", 
                        "all_not_ger", "male_not_ger", "female_not_ger"]

# Print head of data to check if the data is valid.
#print(student_data.head())
#print(student_data.describe())

# Calculate the number of missing data values in each column.
student_data_missing = student_data.apply(lambda x: sum(x.isnull()), axis=0)

for i in student_data_missing:
    if student_data_missing[i] != 0:
        print(student_data_missing[i], "Values are Missing in column:", i)
    

# Convert pandas dataframe columns into numpy arrays
all_students_vec = student_data["all"].to_numpy()
year_vec = student_data["year"].to_numpy()

# Fitting with linear regression
mx, my, bx, by = linear_regression(year_vec, all_students_vec)

lin_reg_x = []
lin_reg_y = []

for x in year_vec:
    lin_reg_x.append(linear_function(mx, x, bx))

for y in all_students_vec:
    lin_reg_y.append(linear_function(my, y, by))

# Prediction of the number of students in the future with linear regresssion.
prediction_year = [i for i in range(2017,2030)]
predictions = lin_predict(prediction_year, year_vec, all_students_vec)[0]


# Tensorflow model for better prediction
# Prediction range
future_year = [i for i in range(2017,2030)]

# Normalize data
all_students_norm = normalize_minmax(all_students_vec, 0, 500000, 0, 1)
year_norm = normalize_minmax(year_vec, 1990, 2050, 0, 1)
future_year_norm = normalize_minmax(future_year, 1990, 2050, 0, 1)


# Split data in training and test data
train_x, test_x = split_data_train_test(year_norm, 0.67)
train_y, test_y = split_data_train_test(all_students_norm, 0.67)

print(np.shape(train_x))
print(train_x)
test = [train_x]

print(test)
print(np.shape(test))


# reshape input to be [samples, time steps, features]
#train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
#test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
"""
# Setting up machine learning model with keras
#model_1 = keras.Sequential()
#model_1.add(keras.layers.Dense(1024, input_dim=1, activation="relu"))
#model_1.add(keras.layers.Dense(1, activation="sigmoid"))
#model_1.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])

# Alternative model 
#model_2 = keras.Sequential()
#model_2.add(LSTM(32, input_shape=(20, 1), return_sequences=False))
#model_2.add(Dense(1))
#model_2.compile(loss='mse', optimizer='adam')

look_back = 1
model_3 = keras.Sequential() 
model_3.add(keras.layers.LSTM(4, input_shape=(1, look_back))) 
model_3.add(keras.layers.Dense(1)) 
model_3.compile(loss='mean_squared_error', optimizer='adam') 


# Train the model
#model_1.fit(year_norm, all_students_norm, epochs=100)
#model_2.fit(year_norm, all_students_norm, batch_size = 2, epochs=100) 
model_3.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2) 

# Calculate predictions
#tf_predictions_1 = model_1.predict(future_year_norm)
#tf_predictions_2 = model_2.predict(future_year_norm)
#tf_predictions_transpose = np.array(de_normalize_minmax(tf_predictions_1, all_students_vec, 0, 500000, 0, 1))
#print(tf_predictions_1)
#print(tf_predictions_transpose.flatten())


# Plot the evolution of the number of students over time.
plt.plot(year_vec, all_students_vec, "-b", label="given data")
plt.plot(year_vec, lin_reg_x, "-r", label="linear regression")
plt.plot(prediction_year, predictions, "--r", label="linear prediction")
plt.plot(future_year, tf_predictions_transpose.flatten(), ":g", label="tensorflow")

plt.title("New students at Bavarian universities since 1998")
plt.xlabel("Year")
plt.ylabel("Number of students")
plt.legend()
plt.show()
"""
