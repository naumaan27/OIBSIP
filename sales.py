# -----------------------
# Sales Prediction Script
# -----------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

df = pd.read_csv(r"C:\Users\nauma\Downloads\archive\Advertising.csv")  

print("\n Dataset Head:\n", df.head())
print("\n Null values:\n", df.isnull().sum())
print("\n Correlation:\n", df.corr())

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

print("\nEnter your own Ad budget to predict sales:")
tv = float(input("TV Spend: ₹"))
radio = float(input("Radio Spend: ₹"))
news = float(input("Newspaper Spend: ₹"))

user_input = pd.DataFrame([[tv, radio, news]], columns=['TV', 'Radio', 'Newspaper'])
user_pred = model.predict(user_input)[0]
print(f"\nPredicted Sales: {user_pred:.2f} units")
