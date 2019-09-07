# students.py

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def linear_regression (x_data, y_data):
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
    return m * x + b

# Read the data form the input file and name the columns. 
student_data = pd.read_csv("student_data.txt", sep=" ", header=None)
student_data.columns = ["semester", "year", "all", "all_male", "all_female", "all_ger", "male_ger", "female_ger", "all_not_ger", "male_not_ger", "female_not_ger"]

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
    print(x)
    lin_reg_x.append(linear_function(mx, x, bx))

for y in all_students_vec:
    print(y)
    lin_reg_y.append(linear_function(my, y, by))

# Plot the evolution of the number of students over time
plt.plot(year_vec, all_students_vec,
    year_vec, lin_reg_x,
    year_vec, lin_reg_y)

plt.xlabel('Years')
plt.ylabel('Number of students')
plt.show()

