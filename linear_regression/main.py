import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/weather.csv')


X = df[['MinTemp', 'MaxTemp', 'AvgHumidity', 'AvgPressure']]
y = df['MaxTemp']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression MSE: {mse:.2f}")
print(f"Linear Regression R^2: {r2:.2f}")


results = pd.DataFrame({'Gerçek': y_test, 'Tahmin': y_pred})
results.to_csv('results.csv', index=False)
