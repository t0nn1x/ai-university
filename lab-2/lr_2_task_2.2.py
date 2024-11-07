import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
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
X_features = X_encoded[:, :-1].astype(float)
y = X_encoded[:, -1].astype(int)

# Збереження LabelEncoder для цільової змінної
target_label_encoder = label_encoders[n_features - 1]

# Розбивка даних на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=5)

# Словник для збереження показників
metrics = {}

# 1. Поліноміальне ядро
poly_classifier = OneVsOneClassifier(SVC(kernel='poly', degree=3, random_state=0))
poly_classifier.fit(X_train, y_train)
y_test_pred = poly_classifier.predict(X_test)

# Обчислення показників
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['Поліноміальне ядро'] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# 2. Гауссове ядро (RBF)
rbf_classifier = OneVsOneClassifier(SVC(kernel='rbf', random_state=0))
rbf_classifier.fit(X_train, y_train)
y_test_pred = rbf_classifier.predict(X_test)

# Обчислення показників
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['Гауссове ядро (RBF)'] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# 3. Сигмоїдальне ядро
sigmoid_classifier = OneVsOneClassifier(SVC(kernel='sigmoid', random_state=0))
sigmoid_classifier.fit(X_train, y_train)
y_test_pred = sigmoid_classifier.predict(X_test)

# Обчислення показників
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

metrics['Сигмоїдальне ядро'] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Виведення показників для кожного класифікатора
for kernel_type, scores in metrics.items():
    print(f"\nПоказники для класифікатора з {kernel_type}:")
    print(f"Accuracy: {round(scores['Accuracy'] * 100, 2)}%")
    print(f"Precision: {round(scores['Precision'] * 100, 2)}%")
    print(f"Recall: {round(scores['Recall'] * 100, 2)}%")
    print(f"F1 Score: {round(scores['F1 Score'] * 100, 2)}%")
