import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print("diabetes.data.shape=",diabetes.data.shape)
print("dir(diabetes)",dir(diabetes))
print("diabetes.target.shape=",diabetes.target.shape)
try:
  print("diabetes.feature_names=",diabetes.feature_names)
except:
  print("No diabetes.feature_names=")


import xlsxwriter
import pandas as pd

try:
  df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
except:
  df = pd.DataFrame(diabetes.data, columns= ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

df['target'] = diabetes.target


print(df.head())
df.to_csv("diabetes.csv", sep='\t')
writer = pd.ExcelWriter('diabetes.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.show()