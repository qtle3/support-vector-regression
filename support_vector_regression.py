# Support Vector Regression

# import libraries
import numpy as np
import pandas as pd
import matplotlib as plt

# import sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(dataset)
y = y.reshape(-1, 1)


# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)
print(y)

# Training the SVR model with the Gaussian Radial Basis Function Kernal on the whole dataset
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Reverse Scaling and Predict a new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))
