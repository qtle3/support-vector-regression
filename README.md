# Salary Prediction using Support Vector Regression (SVR)

This project implements **Support Vector Regression (SVR)** to predict the salary of an employee based on their position level. The script uses a dataset that contains position levels and corresponding salaries, applying feature scaling to improve the performance of the SVR model. The results are visualized to show how well the model fits the data.

## Detailed Summary

The script loads the `Position_Salaries.csv` dataset, containing employee position levels and their respective salaries. **Support Vector Regression** is trained on the dataset using the **Gaussian Radial Basis Function (RBF) Kernel**, which is ideal for capturing non-linear relationships in the data. The script applies **feature scaling** to both the input features (position levels) and the target variable (salaries) before training the SVR model.
