import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Veriyi oku
url = 'https://raw.githubusercontent.com/z-ozturk/python_weather_prediction/main/data/weather.csv'
df = pd.read_csv(url)

# Sadece ihtiyacın olan sütunları al
df = df[['DateTime', 'MaxTemp', 'MinTemp', 'AvgHumidity', 'AvgPressure']]

# Tarih sütununu datetime'a çevir ve indeks yap
df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True)
df.set_index('DateTime', inplace=True)

# Eksik veri kontrolü
df.dropna(inplace=True)

# Özellikler ve hedef
X = df[['MinTemp', 'AvgHumidity', 'AvgPressure']]
y = df['MaxTemp']

# Veriyi train/test olarak böl (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Test üzerinde tahmin yap
y_pred = model.predict(X_test)

# Performans ölçümleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ortalama Kare Hata (MSE): {mse:.2f}")
print(f"R Kare (R²) Skoru: {r2:.2f}")


# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Doğrusal referans çizgisi
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Değerleri')
plt.title('Gerçek vs. Tahmin Değerleri')
plt.grid(True)
plt.show()


fark = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.plot(fark.values, marker='o', linestyle='-', color='orange')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Tahmin Hataları (Gerçek - Tahmin)')
plt.xlabel('Örnek')
plt.ylabel('Hata')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(fark, bins=20, edgecolor='black', color='skyblue')
plt.title('Tahmin Hatalarının Dağılımı')
plt.xlabel('Hata (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()
