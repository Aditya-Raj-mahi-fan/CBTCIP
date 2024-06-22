import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# Load the data with the correct column names
file_path = r"C:\Users\adira\OneDrive\Desktop\Miles_Traveled.csv"
df = pd.read_csv(file_path, index_col='DATE', parse_dates=True)

# Rename the value column to a simpler name for convenience
df.rename(columns={'TRFVOLUSM227NFWA': 'value'}, inplace=True)

# Display the first few rows of the dataframe
print(df.head())

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit the ARIMA model
model = ARIMA(train['value'], order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
start = len(train)
end = len(train) + len(test) - 1
predictions = model_fit.predict(start=start, end=end, typ='levels')

# Evaluate the model using RMSE
rmse = sqrt(mean_squared_error(test['value'], predictions))
print(f'RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Test')
plt.plot(test.index, predictions, label='Predicted')
plt.legend()
plt.show()
