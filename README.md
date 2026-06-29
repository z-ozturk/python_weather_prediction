# Hava Durumu Tahmini / Weather Prediction

## TR — Türkçe

### Proje Özeti

Bu proje, İstanbul'a ait 2009–2019 yılları arasındaki günlük hava durumu verilerini kullanarak makine öğrenmesi modelleriyle maksimum sıcaklık (`MaxTemp`) tahmini yapar.

**Özellikler (girdi):** `MinTemp`, `AvgHumidity`, `AvgPressure`  
**Hedef (çıktı):** `MaxTemp` (°C)

İki farklı script ve bir Jupyter not defteri içerir:

| Dosya | Açıklama |
|---|---|
| `linear_regression/main.py` | Doğrusal Regresyon modeli — MSE/R² ölçümleri ve 3 grafik |
| `multiple_models/compare_models.py` | 7 farklı modeli karşılaştırır — RMSE ve R² sonuçları |
| `linear_regression/WeatherPrediction_LinearRegression.ipynb` | Adım adım açıklamalı Jupyter not defteri |

### Gereksinimler

- Python 3.8 veya üzeri
- pip

### Kurulum

```bash
# 1. Depoyu klonlayın
git clone https://github.com/z-ozturk/python_weather_prediction.git
cd python_weather_prediction

# 2. Sanal ortam oluşturun ve etkinleştirin
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Çalıştırma

**Doğrusal Regresyon:**

```bash
python linear_regression/main.py
```

**7 Modeli Karşılaştır:**

```bash
python multiple_models/compare_models.py
```

**Jupyter Not Defteri (opsiyonel):**

```bash
pip install notebook
jupyter notebook linear_regression/WeatherPrediction_LinearRegression.ipynb
```

### Beklenen Çıktı

`linear_regression/main.py` çalıştırıldığında terminalde şu çıktı görünür:

```
Ortalama Kare Hata (MSE): 5.04
R Kare (R²) Skoru: 0.92
```

Ardından sırayla 3 matplotlib penceresi açılır:
1. **Gerçek vs. Tahmin** — dağılım grafiği (scatter plot)
2. **Tahmin Hataları** — örnek başına hata değerleri
3. **Hata Dağılımı** — histogram

`multiple_models/compare_models.py` çalıştırıldığında 7 model için RMSE ve R² değerleri yazdırılır:

```
Linear Regression: RMSE = 2.25, R² = 0.91
Decision Tree:     RMSE = 2.49, R² = 0.89
Random Forest:     RMSE = 1.98, R² = 0.93
Gradient Boosting: RMSE = 2.02, R² = 0.93
KNN:               RMSE = 2.31, R² = 0.91
SVR:               RMSE = 2.61, R² = 0.88
Neural Network:    RMSE = 2.18, R² = 0.92
```

> Not: `compare_models.py` herhangi bir grafik açmaz; sonuçlar yalnızca terminale yazdırılır.

### Veri Seti

Scriptler veriyi doğrudan bu deponun GitHub ham URL'sinden çeker — `data/weather.csv` dosyasını ayrıca indirmenize gerek yoktur. `data/weather.csv` referans amaçlı depoda bulunmaktadır.

Veri seti toplam ~3.900 günlük kayıt içerir. Kullanılan sütunlar:

| Sütun | Açıklama |
|---|---|
| `DateTime` | Tarih (GG.AA.YYYY) |
| `MaxTemp` | Günlük maksimum sıcaklık (°C) — **tahmin hedefi** |
| `MinTemp` | Günlük minimum sıcaklık (°C) |
| `AvgHumidity` | Ortalama nem (%) |
| `AvgPressure` | Ortalama hava basıncı (hPa) |

---

## EN — English

### Project Summary

This project predicts daily maximum temperature (`MaxTemp`) for Istanbul using historical weather data from 2009 to 2019.

**Features (input):** `MinTemp`, `AvgHumidity`, `AvgPressure`  
**Target (output):** `MaxTemp` (°C)

| File | Description |
|---|---|
| `linear_regression/main.py` | Linear Regression model — prints MSE/R² and shows 3 charts |
| `multiple_models/compare_models.py` | Compares 7 models — prints RMSE and R² for each |
| `linear_regression/WeatherPrediction_LinearRegression.ipynb` | Step-by-step annotated Jupyter notebook |

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/z-ozturk/python_weather_prediction.git
cd python_weather_prediction

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### How to Run

**Linear Regression:**

```bash
python linear_regression/main.py
```

**Compare 7 Models:**

```bash
python multiple_models/compare_models.py
```

**Jupyter Notebook (optional):**

```bash
pip install notebook
jupyter notebook linear_regression/WeatherPrediction_LinearRegression.ipynb
```

### Expected Output

Running `linear_regression/main.py` prints:

```
Ortalama Kare Hata (MSE): 5.04
R Kare (R²) Skoru: 0.92
```

Then opens 3 matplotlib windows in sequence:
1. **Actual vs. Predicted** — scatter plot
2. **Prediction Errors** — error value per sample
3. **Error Distribution** — histogram

Running `multiple_models/compare_models.py` prints RMSE and R² for all 7 models:

```
Linear Regression: RMSE = 2.25, R² = 0.91
Decision Tree:     RMSE = 2.49, R² = 0.89
Random Forest:     RMSE = 1.98, R² = 0.93
Gradient Boosting: RMSE = 2.02, R² = 0.93
KNN:               RMSE = 2.31, R² = 0.91
SVR:               RMSE = 2.61, R² = 0.88
Neural Network:    RMSE = 2.18, R² = 0.92
```

> Note: `compare_models.py` produces no charts — results are printed to the terminal only.

### Dataset

Both scripts fetch data directly from this repository's raw GitHub URL — you do not need to download `data/weather.csv` manually. The file is included in the repository for reference.

The dataset contains ~3,900 daily records. Columns used by the models:

| Column | Description |
|---|---|
| `DateTime` | Date (DD.MM.YYYY) |
| `MaxTemp` | Daily maximum temperature (°C) — **prediction target** |
| `MinTemp` | Daily minimum temperature (°C) |
| `AvgHumidity` | Average humidity (%) |
| `AvgPressure` | Average air pressure (hPa) |
