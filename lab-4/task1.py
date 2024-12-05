import numpy as np
import matplotlib.pyplot as plt

# Введення даних
X = np.array([8, 7, 6, 5, 4, 3])
Y = np.array([2, 4, 6, 8, 10, 12])

# Сортування даних за зростанням X для зручності побудови графіку
sorted_indices = np.argsort(X)
X = X[sorted_indices]
Y = Y[sorted_indices]

# Використання методу найменших квадратів для знаходження коефіцієнтів
coefficients = np.polyfit(X, Y, 1)  # Ступінь 1 для лінійної функції
a, b = coefficients

print(f"Знайдені коефіцієнти: a = {a}, b = {b}")

# Створення лінійного простору для X
X_line = np.linspace(min(X), max(X), 100)
Y_line = a * X_line + b

# Побудова графіку
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Експериментальні дані')
plt.plot(X_line, Y_line, color='red', label='Апроксимуюча функція')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Метод найменших квадратів')
plt.legend()
plt.grid(True)
plt.show()
