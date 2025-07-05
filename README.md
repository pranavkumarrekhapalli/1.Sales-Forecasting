# 1.Sales-Forecasting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_excel("sample_sales_data.xlsx")
df['date'] = pd.to_datetime(df['date'])
daily_sales = df.groupby('date').agg({'quantity': 'sum', 'revenue': 'sum'}).reset_index()
daily_sales['days_since_start'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
X = daily_sales[['days_since_start']]
y = daily_sales['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
plot_df = X_test.copy()
plot_df['Actual'] = y_test
plot_df['Predicted'] = y_pred
plot_df = plot_df.sort_values('days_since_start')

plt.figure(figsize=(10, 6))
plt.plot(plot_df['days_since_start'], plot_df['Actual'], label='Actual Revenue', marker='o')
plt.plot(plot_df['days_since_start'], plot_df['Predicted'], label='Predicted Revenue', linestyle='--')
plt.xlabel("Days Since Start")
plt.ylabel("Revenue")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.grid(True)
plt.show()
last_day = daily_sales['days_since_start'].max()
future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
future_preds = model.predict(future_days)
forecast_df = pd.DataFrame({
    'Day': future_days.flatten(),
    'Predicted_Revenue': future_preds
})
print(forecast_df)
