# Support Vector Regression

# import libraries
import numpy as np
import pandas as pd
import matplotlib as plt

# import sklearn libraries
from sklearn.preprocessing import StandardScaler


# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(dataset)
y = y.reshape(-1, 1)


# Feature Scaling
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

print(X)
print(y)
