# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Load the dataset from a CSV file
data = pd.read_csv('data/house_data.csv')

# Display the first few rows of the dataset to get an overview
print(data.head())

# Splitting data into features (X) and target (y) variables
# 'area' and 'rooms' are our features, while 'price' is the target variable
X = data[['LotFrontage', 'LotArea']]
y = data['SalePrice']

# Dividing the dataset into a training set and a testing set
# 80% of the data will be used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Using the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Plotting the actual prices vs the predicted prices to visualize the performance
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Evaluating the model's performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Displaying the coefficients of the features and the model's intercept
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
