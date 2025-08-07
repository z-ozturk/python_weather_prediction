import pandas as pd
import matplotlib.pyplot as plt


results = pd.read_csv('results.csv')

# Scatter plot: Gerçek vs Tahmin
plt.figure(figsize=(8, 6))
plt.scatter(results['Gerçek'], results['Tahmin'], alpha=0.7)
plt.plot([results['Gerçek'].min(), results['Gerçek'].max()],
         [results['Gerçek'].min(), results['Gerçek'].max()], 'r--')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')
plt.title('Linear Regression: Gerçek vs Tahmin')
plt.grid(True)
plt.show()


# Hata grafiği (fark grafiği)
results['Hata'] = results['Tahmin'] - results['Gerçek']

plt.figure(figsize=(8, 4))
plt.plot(results['Hata'].values, marker='o', linestyle='', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Örnek indeksi')
plt.ylabel('Hata (Tahmin - Gerçek)')
plt.title('Tahmin Hatalarının Dağılımı')
plt.grid(True)
plt.show()


# Hata histogramı
plt.figure(figsize=(8, 4))
plt.hist(results['Hata'], bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Hata (Tahmin - Gerçek)')
plt.ylabel('Frekans')
plt.title('Tahmin Hatalarının Histogramı')
plt.grid(True)
plt.show()
