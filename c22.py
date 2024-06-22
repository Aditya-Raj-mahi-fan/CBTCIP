import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Load the data
file_path = r"C:\Users\adira\OneDrive\Desktop\Alcohol_Sales.csv"
alcohol_sales_df = pd.read_csv(file_path, index_col='DATE', parse_dates=True)

# Display the first few rows and column names of the dataframe to inspect the column names
print(alcohol_sales_df.head())
print(alcohol_sales_df.columns)

# Rename the value column if necessary (assuming the value column is named 'Sales')
# Update the column name based on the actual column names in your file
# For this example, we'll assume it's 'S4248SM144NCEN'
alcohol_sales_df.rename(columns={'S4248SM144NCEN': 'alcohol_sales'}, inplace=True)

# Display the first few rows again to confirm the renaming
print(alcohol_sales_df.head())

# Split the data into training and testing sets
train_size = int(len(alcohol_sales_df) * 0.8)
train_alcohol, test_alcohol = alcohol_sales_df[:train_size], alcohol_sales_df[train_size:]

# Fit the ARIMA model
alcohol_sales_model = ARIMA(train_alcohol['alcohol_sales'], order=(5, 1, 0))
alcohol_sales_model_fit = alcohol_sales_model.fit()

# Make predictions
start = len(train_alcohol)
end = len(train_alcohol) + len(test_alcohol) - 1
alcohol_sales_predictions = alcohol_sales_model_fit.predict(start=start, end=end, typ='levels')

# Evaluate the model using RMSE
alcohol_sales_rmse = sqrt(mean_squared_error(test_alcohol['alcohol_sales'], alcohol_sales_predictions))
print(f'RMSE: {alcohol_sales_rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_alcohol.index, train_alcohol['alcohol_sales'], label='Train Alcohol Sales')
plt.plot(test_alcohol.index, test_alcohol['alcohol_sales'], label='Test Alcohol Sales')
plt.plot(test_alcohol.index, alcohol_sales_predictions, label='Predicted Alcohol Sales')
plt.legend()
plt.show()
