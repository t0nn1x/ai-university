from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Завантаження даних Iris
iris = load_iris()
X = iris.data
y = iris.target

# Створення моделі k-means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, random_state=42)

# Навчання моделі
kmeans.fit(X)

# Отримання прогнозів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів (для перших двох ознак)
plt.figure(figsize=(12, 5))

# Графік розсіювання з кластерами
plt.subplot(1, 2, 1)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=100, c='red', label='Кластер 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=100, c='blue', label='Кластер 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=100, c='green', label='Кластер 3')

# Відображення центроїдів
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', marker='*', label='Центроїди')

plt.title('Кластеризація ірисів (перші дві ознаки)')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.legend()

# Графік розсіювання для інших двох ознак
plt.subplot(1, 2, 2)
plt.scatter(X[y_kmeans == 0, 2], X[y_kmeans == 0, 3],
            s=100, c='red', label='Кластер 1')
plt.scatter(X[y_kmeans == 1, 2], X[y_kmeans == 1, 3],
            s=100, c='blue', label='Кластер 2')
plt.scatter(X[y_kmeans == 2, 2], X[y_kmeans == 2, 3],
            s=100, c='green', label='Кластер 3')

# Відображення центроїдів
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
            s=200, c='yellow', marker='*', label='Центроїди')

plt.title('Кластеризація ірисів (останні дві ознаки)')
plt.xlabel('Довжина пелюстки')
plt.ylabel('Ширина пелюстки')
plt.legend()

plt.tight_layout()
plt.show()

# Оцінка якості кластеризації
print("\nРезультати кластеризації:")
print(f"Інерція: {kmeans.inertia_:.2f}")
print(f"Коефіцієнт силуету: {silhouette_score(X, y_kmeans):.2f}")

# Порівняння з реальними мітками
print(
    f"Індекс Ренда (порівняння з реальними мітками): {adjusted_rand_score(y, y_kmeans):.2f}")

# Виведення центрів кластерів
print("\nЦентри кластерів:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Кластер {i+1}:", end=" ")
    print(f"Довжина чашолистка: {center[0]:.2f}, ", end="")
    print(f"Ширина чашолистка: {center[1]:.2f}, ", end="")
    print(f"Довжина пелюстки: {center[2]:.2f}, ", end="")
    print(f"Ширина пелюстки: {center[3]:.2f}")
