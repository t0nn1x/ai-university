import numpy as np
from sklearn import preprocessing

# Нові дані input_data
input_data = np.array([
    [-3.3, -1.6, 6.1],
    [2.4, -1.2, 4.3],
    [-3.2, 5.5, -6.1],
    [-4.4, 1.4, -1.2]
])

# Бінаризація даних з порогом 2.1
binarizer = preprocessing.Binarizer(threshold=2.1)
data_binarized = binarizer.transform(input_data)
print("\nБінаризовані дані:\n", data_binarized)

# Виведення середнього значення та стандартного відхилення перед масштабуванням
print("\nBEFORE (Перед масштабуванням):")
print("Середнє =", input_data.mean(axis=0))
print("Стандартне відхилення =", input_data.std(axis=0))

# Виключення середнього значення (масштабування)
data_scaled = preprocessing.scale(input_data)
print("\nAFTER (Після масштабування):")
print("Середнє =", data_scaled.mean(axis=0))
print("Стандартне відхилення =", data_scaled.std(axis=0))

# Масштабування MinMax
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = min_max_scaler.fit_transform(input_data)
print("\nМасштабовані дані MinMax:\n", data_scaled_minmax)

# Нормалізація даних (l1 та l2)
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nНормалізовані дані (l1):\n", data_normalized_l1)
print("\nНормалізовані дані (l2):\n", data_normalized_l2)
