# Salary Prediction using Support Vector Regression (SVR)

This project implements **Support Vector Regression (SVR)** to predict the salary of an employee based on their position level. The script uses a dataset that contains position levels and corresponding salaries, applying feature scaling to improve the performance of the SVR model. The results are visualized to show how well the model fits the data.

## Detailed Summary

The script loads the `Position_Salaries.csv` dataset, containing employee position levels and their respective salaries. **Support Vector Regression** is trained on the dataset using the **Gaussian Radial Basis Function (RBF) Kernel**, which is ideal for capturing non-linear relationships in the data. The script applies **feature scaling** to both the input features (position levels) and the target variable (salaries) before training the SVR model.

The script performs the following steps:

1. **Data Preprocessing:**
   - Loads the dataset and separates it into independent variable (position level) and dependent variable (salary).
   - Reshapes the target variable `y` to a 2D array to ensure compatibility with the scaling process.

2. **Feature Scaling:**
   - Applies **StandardScaler** to scale the features and target variable, which is crucial for SVR to perform optimally as it is sensitive to the range of input data.

3. **SVR Model Training:**
   - Trains the **SVR model** with a **Radial Basis Function (RBF) kernel** on the scaled data.
   
4. **Prediction:**
   - Makes predictions on new data points (position level 6.5) by reversing the scaling transformation to interpret the predicted values in the original salary scale.

5. **Visualization:**
   - The script visualizes the SVR results by plotting both the original and predicted salaries, showing the fit of the model to the dataset.
   - A higher resolution plot is also generated to display a smoother curve that captures the non-linearity of the data.
