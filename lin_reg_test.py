# lin_reg_test.py

import numpy as np
import matplotlib.pyplot as plt

def linear_regression (x_data, y_data):
    x_mean = sum(x_data) / np.prod(x_data.shape)
    y_mean = sum(y_data) / np.prod(y_data.shape)

    print("X mean:", x_mean, "Sum X =", sum(x_data))
    print("Y mean:", y_mean, "Sum Y =", sum(y_data))

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


x_data = np.array([15, 3, 7.5, 12.5, 56, 34.5, 12.8, 56.8, 25.4, 29])
y_data = np.array([16.3, 4, 14.5, 15, 58, 40, 22, 65, 19, 30])

# Fitting with linear regression
mx, my, bx, by = linear_regression(x_data, y_data)

lin_reg_x = []
lin_reg_y = []

for x in x_data:
    lin_reg_x.append(linear_function(mx, x, bx))

for y in y_data:
    lin_reg_y.append(linear_function(my, y, by))

# Plotting results
plt.plot(x_data, y_data, "o",
    x_data, lin_reg_x,
    y_data, lin_reg_y)

plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()