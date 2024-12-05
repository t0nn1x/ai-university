import numpy as np
import matplotlib.pyplot as plt

# Крок 1: Вихідні дані
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

# Заповнення матриці Вандермонда для полінома степеня 4
X = np.vander(x, N=5, increasing=False)
print("Матриця Вандермонда X:")
print(X)

# Крок 2: Отримання коефіцієнтів інтерполяційного полінома
coefficients = np.linalg.solve(X, y)
print("\nКоефіцієнти інтерполяційного полінома:")
print(coefficients)

# Крок 3: Визначення функції полінома
polynomial = np.poly1d(coefficients)
print("\nІнтерполяційний поліном:")
print(polynomial)

# Крок 4: Побудова графіка функції
x_plot = np.linspace(min(x) - 0.1, max(x) + 0.1, 500)
y_plot = polynomial(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='Інтерполяційний поліном', color='blue')
plt.scatter(x, y, color='red', label='Вихідні точки')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяція поліномом степеня 4')
plt.legend()
plt.grid(True)
plt.show()

# Крок 5: Визначення значення функції в проміжних точках
x_values = [0.2, 0.5]
y_values = polynomial(x_values)

for x_val, y_val in zip(x_values, y_values):
    print(f"\nЗначення полінома в точці x = {x_val}: y = {y_val}")
