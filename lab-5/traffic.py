import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split  # Замінено cross_validation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder

# Завантаження даних із файлу traffic_data.txt
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(data.shape)

# Кодування кожної колонки окремо
for i, item in enumerate(data[0]):
    if item.isdigit():  # Якщо значення числове
        X_encoded[:, i] = data[:, i]
    else:  # Якщо значення рядкове
        label_encoder.append(LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

# Конвертація в числовий тип
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Створення та навчання регресора
params = {
    'n_estimators': 100,
    'max_depth': 4,
    'random_state': 0
}

regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Обчислення показників ефективності на тестових даних
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Тестування на конкретному прикладі
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)

# Кодування тестової точки
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        test_datapoint_encoded[i] = \
            int(label_encoder[count].transform([test_datapoint[i]])[0])
        count = count + 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозування результату
print("\nPredicted traffic:",
      int(regressor.predict([test_datapoint_encoded])[0]))
