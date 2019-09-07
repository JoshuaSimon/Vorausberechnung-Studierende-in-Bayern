# students.py

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

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

# Plot the evolution of the number of students over time
plt.plot(year_vec, all_students_vec)
plt.xlabel('Years')
plt.ylabel('Number of students')
plt.show()

