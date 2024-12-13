import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантаження даних California Housing замість Boston
housing_data = fetch_california_housing()

# Перемішування даних
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# Модель на основі регресора AdaBoost
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)
regressor.fit(X_train, y_train)

# Обчислення показників ефективності регресора AdaBoost
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Вилучення важливості ознак
feature_importances = regressor.feature_importances_
# Перетворення в numpy array
feature_names = np.array(housing_data.feature_names)

# Нормалізація значень важливості ознак
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування та перестановка значень
index_sorted = np.argsort(feature_importances)[
    ::-1]  # Змінений спосіб сортування

# Розміщення міток уздовж осі X
pos = np.arange(len(feature_names))

# Побудова стовпчастої діаграми
plt.figure(figsize=(12, 6))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted], rotation=45, ha='right')
plt.ylabel('Відносна важливість (%)')
plt.xlabel('Ознаки')
plt.title('Оцінка важливості ознак з використанням регресора AdaBoost')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Виведення важливості ознак у відсотках
print("\nВажливість ознак у відсотках:")
for idx in index_sorted:
    print(f"{feature_names[idx]}: {feature_importances[idx]:.2f}%")
