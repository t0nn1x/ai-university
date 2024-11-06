import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Відкриття файлу та читання рядків
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        # Розбиття рядка на дані
        data = line[:-1].split(', ')
        # Перевірка мітки класу і додавання до відповідного класу
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Індекси категоріальних ознак (0-based indexing)
categorical_feature_indices = [1, 3, 5, 6, 7, 8, 9, 13]

# Перетворення рядкових даних на числові
label_encoders = {}
X_encoded = np.empty(X.shape)
n_samples, n_features = X.shape

for i in range(n_features):
    feature = X[:, i]
    if i in categorical_feature_indices or i == n_features - 1:  # Включаючи цільову змінну
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(feature)
        label_encoders[i] = le
    else:
        X_encoded[:, i] = feature.astype(int)

# Розділення даних на X_features і y
X_features = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Збереження LabelEncoder для цільової змінної
target_label_encoder = label_encoders[n_features - 1]

# Створення SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Розбивка даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=5)

# Навчання класифікатора на тренувальних даних
classifier.fit(X_train, y_train)

# Прогнозування на тестових даних
y_test_pred = classifier.predict(X_test)

# Обчислення показників якості класифікації
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

# Обчислення F1-міри за допомогою перехресної перевірки
f1_scores = cross_val_score(classifier, X_features, y,
                            scoring='f1_weighted', cv=3)
print("Cross-validated F1 score:", round(100 * f1_scores.mean(), 2), "%")

# Тестова точка даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = np.array([-1] * len(input_data))
for i, item in enumerate(input_data):
    if i in categorical_feature_indices:
        le = label_encoders[i]
        input_data_encoded[i] = le.transform([item])[0]
    else:
        input_data_encoded[i] = int(item)

# Прогнозування результату для тестової точки даних
predicted_class = classifier.predict([input_data_encoded])

# Виведення результату
predicted_label = target_label_encoder.inverse_transform(predicted_class)[0]
print("Результат класифікації для тестової точки даних:", predicted_label)
