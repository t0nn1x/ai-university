from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Завантаження даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна для X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
print(f"\nОцінена ширина вікна: {bandwidth_X:.2f}")

# Створення та навчання моделі Mean Shift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Отримання міток кластерів та центрів
labels = meanshift_model.labels_
cluster_centers = meanshift_model.cluster_centers_
n_clusters = len(np.unique(labels))

print(f"\nКількість знайдених кластерів: {n_clusters}")
print("\nКоординати центрів кластерів:")
for i, center in enumerate(cluster_centers):
    print(f"Кластер {i + 1}: {center}")

# Візуалізація результатів
plt.figure(figsize=(12, 5))

# Графік вхідних даних
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=50)
plt.title('Вхідні дані')
plt.xlabel('X')
plt.ylabel('Y')

# Графік результатів кластеризації
plt.subplot(1, 2, 2)
colors = cycle('bgrcmyk')
for k, col in zip(range(n_clusters), colors):
    cluster_members = labels == k
    cluster_data = X[cluster_members]
    plt.plot(cluster_data[:, 0], cluster_data[:, 1], col + '.',
             markersize=10, label=f'Кластер {k+1}')

    # Відображення центру кластера
    center = cluster_centers[k]
    plt.plot(center[0], center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=15)

plt.title(f'Результат кластеризації\n{n_clusters} кластерів знайдено')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()

# Оцінка якості кластеризації
if n_clusters > 1:  # Silhouette score потребує мінімум 2 кластери
    silhouette_avg = silhouette_score(X, labels)
    print(f"\nSilhouette score: {silhouette_avg:.3f}")

# Додатковий аналіз розподілу точок по кластерах
print("\nРозподіл точок по кластерах:")
for i in range(n_clusters):
    cluster_size = np.sum(labels == i)
    print(f"Кластер {i + 1}: {cluster_size} точок")

# Обчислення внутрішньокластерної дисперсії
inertia = 0
for i in range(n_clusters):
    cluster_points = X[labels == i]
    center = cluster_centers[i]
    inertia += np.sum((cluster_points - center) ** 2)
print(f"\nВнутрішньокластерна дисперсія: {inertia:.2f}")
