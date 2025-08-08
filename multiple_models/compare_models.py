import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Veriyi oku
url = 'https://raw.githubusercontent.com/z-ozturk/python_weather_prediction/main/data/weather.csv'
df = pd.read_csv(url)

# Sadece ihtiyacın olan sütunları al
df = df[['DateTime', 'MaxTemp', 'MinTemp', 'AvgHumidity', 'AvgPressure']]

# Tarih sütununu datetime'a çevir ve indeks yap
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True)
df.set_index('DateTime', inplace=True)

# Eksik veri temizle
df.dropna(inplace=True)

# Özellikler ve hedef
X = df[['MinTemp', 'AvgHumidity', 'AvgPressure']]
y = df['MaxTemp']

# Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kullanılacak modeller
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Neural Network": MLPRegressor(max_iter=1000)
}

# Modelleri dene
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE = {rmse:.2f}, R² = {r2:.2f}")
