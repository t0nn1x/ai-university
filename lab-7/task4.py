import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import affinity_propagation
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Визначення компаній технологічного сектору
company_symbols_map = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "IBM": "IBM Corp.",
    "INTC": "Intel Corp.",
    "ORCL": "Oracle Corp.",
    "CSCO": "Cisco Systems"
}

symbols = list(company_symbols_map.keys())
names = list(company_symbols_map.values())

# Завантаження даних за більш тривалий період
start_date = '2006-01-01'
end_date = '2007-01-01'

# Створення DataFrame для зберігання даних про доходність
returns_data = pd.DataFrame()

# Завантаження та обробка даних
for symbol in symbols:
    try:
        stock = yf.download(symbol, start=start_date, end=end_date)
        if not stock.empty:
            # Розрахунок щоденної доходності
            returns = stock['Close'].pct_change()
            returns_data[symbol] = returns
            print(f"Завантажено та оброблено дані для {symbol}")
    except Exception as e:
        print(f"Помилка завантаження даних для {symbol}: {e}")

# Видалення перших рядків з NaN
returns_data = returns_data.dropna()

# Стандартизація даних
scaler = StandardScaler()
X = scaler.fit_transform(returns_data)

# Створення та навчання моделі з оптимізованими параметрами
edge_model = GraphicalLassoCV(cv=5, max_iter=1000)
edge_model.fit(X)

# Розрахунок матриці кореляції
correlation_matrix = np.corrcoef(X.T)

# Кластеризація з оптимізованими параметрами
_, labels = affinity_propagation(correlation_matrix,
                                 preference=-0.5,  # Зменшуємо преференцію для отримання меншої кількості кластерів
                                 random_state=42)
num_labels = len(set(labels))

# Виведення результатів
print("\nРезультати кластеризації:")
for i in range(num_labels):
    cluster_companies = [names[j]
                         for j, label in enumerate(labels) if label == i]
    print(f"Кластер {i+1} ==> {', '.join(cluster_companies)}")

# Покращена візуалізація
plt.figure(figsize=(15, 10))

# Subplot для матриці кореляції
plt.subplot(2, 1, 1)
plt.imshow(correlation_matrix, cmap='RdYlBu', aspect='auto')
plt.colorbar()
plt.xticks(range(len(names)), names, rotation=45)
plt.yticks(range(len(names)), names)
plt.title('Матриця кореляції між компаніями')

# Subplot для кластерів
plt.subplot(2, 1, 2)
plt.title('Кластеризація компаній на фондовому ринку')

# Покращене відображення зв'язків між компаніями
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if labels[i] == labels[j]:
            plt.plot([i, j], [0, 0], 'k-', linewidth=2, alpha=0.6)

# Відображення компаній з покращеним форматуванням
plt.scatter(range(len(names)), [0] * len(names), c=labels,
            cmap='viridis', s=200, edgecolor='black')

# Додавання підписів компаній
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Збереження результатів
plt.savefig('stock_market_clusters.png', bbox_inches='tight', dpi=300)
plt.close()

# Додатковий аналіз кластерів
print("\nДодатковий аналіз:")
for i in range(num_labels):
    cluster_indices = [j for j, label in enumerate(labels) if label == i]
    cluster_returns = returns_data.iloc[:, cluster_indices]

    print(f"\nКластер {i+1}:")
    print("Компанії:", ", ".join([names[j] for j in cluster_indices]))
    print(f"Середня доходність: {cluster_returns.mean().mean():.4f}")
    print(f"Середня волатильність: {cluster_returns.std().mean():.4f}")

print("\nГрафік збережено у файл 'stock_market_clusters.png'")
