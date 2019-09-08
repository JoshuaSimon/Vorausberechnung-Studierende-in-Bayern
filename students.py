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

    return [mx, my, bx, by]

def linear_function(m, x, b):
    """ A linear function of one variable x 
    with solpe m and and intercept b. 
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
all_students_vec = student_data['all'].to_numpy()
year_vec = student_data['year'].to_numpy()

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


# Plot the evolution of the number of students over time.
plt.plot(year_vec, all_students_vec, "-b", label='given data')
plt.plot(year_vec, lin_reg_x, "-r", label='linear regression')
plt.plot(prediction_year, predictions, "--r", label="linear prediction")

plt.title('Students at Bavarian universities since 1998')
plt.xlabel('Year')
plt.ylabel('Number of students')
plt.legend()
plt.show()


# Tensorflow model for better prediction
# Prediction range
future_year = [i for i in range(2017,2030)]

# Normalize data
all_students_norm = normalize_l2(all_students_vec)
year_norm = normalize_l2(year_vec)
future_year_norm = normalize_l2(future_year)


print(future_year)
print()
print(future_year_norm)
print()
print(de_normalize_l2(future_year_norm, future_year))

# Setting up machine learning model with keras
model = keras.Sequential()
model.add(keras.layers.Dense(1024, input_dim=1, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

# Train the model
#model.fit(year_norm, all_students_norm, epochs=10) 
model.fit(year_vec, all_students_vec, epochs=10) 

# Calculate predictions
tf_predictions = model.predict(future_year_norm)
#print(tf_predictions)
#print(de_normalize_l2(tf_predictions, all_students_vec))

#print(model.predict(normalize_l2([2020])))
#print(model.predict(normalize_l2([2030])))
print(model.predict([2020]))