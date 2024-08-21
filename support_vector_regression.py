# Support Vector Regression

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
regressor.fit(X, y.ravel())

# Reverse Scaling and Predict a new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1, 1))
inverse_X = sc_x.inverse_transform(X)
inverse_y = sc_y.inverse_transform(y)

# Visualizing the SVR Results
plt.scatter(inverse_X, inverse_y, color="red")
plt.plot(
    inverse_X,
    sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
    color="blue",
)
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the SVR Results (for higher resolution and smoother curve)
X_grid = np.arange(inverse_X.min(), inverse_X.max(), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(inverse_X, inverse_y, color="red")
plt.plot(
    X_grid,
    sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1, 1)),
    color="blue",
)
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
