import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data.append(line.strip().split(', '))

# Створення DataFrame
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.DataFrame(data, columns=columns)

# Вибір 25000 зразків кожного класу
df_class_1 = df[df['income'] == '<=50K'].head(25000)
df_class_2 = df[df['income'] == '>50K'].head(25000)
df = pd.concat([df_class_1, df_class_2])

# Розділення ознак та цільової змінної
X = df.drop('income', axis=1)
y = df['income']

# Кодування категоріальних ознак
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_features:
    le = preprocessing.LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Кодування цільової змінної
le_income = preprocessing.LabelEncoder()
y = le_income.fit_transform(y)

# Приведення даних до числового типу
X = X.apply(pd.to_numeric)

# Розбивка даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Визначення моделей
models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Оцінка моделей
seed = 7
scoring = 'accuracy'
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()*100:.2f}% ({cv_results.std()*100:.2f}%)')

# Порівняння моделей
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів')
plt.ylabel('Точність')
plt.show()

# Функція для оцінки моделі
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_pred

# Словник для збереження результатів
model_results = {}

# Логістична регресія
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
accuracy_lr, precision_lr, recall_lr, f1_lr, y_pred_lr = evaluate_model(model_lr, X_test, y_test)
model_results['LR'] = [accuracy_lr, precision_lr, recall_lr, f1_lr]

# Лінійний дискримінантний аналіз
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train, y_train)
accuracy_lda, precision_lda, recall_lda, f1_lda, y_pred_lda = evaluate_model(model_lda, X_test, y_test)
model_results['LDA'] = [accuracy_lda, precision_lda, recall_lda, f1_lda]

# KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
accuracy_knn, precision_knn, recall_knn, f1_knn, y_pred_knn = evaluate_model(model_knn, X_test, y_test)
model_results['KNN'] = [accuracy_knn, precision_knn, recall_knn, f1_knn]

# CART
model_cart = DecisionTreeClassifier()
model_cart.fit(X_train, y_train)
accuracy_cart, precision_cart, recall_cart, f1_cart, y_pred_cart = evaluate_model(model_cart, X_test, y_test)
model_results['CART'] = [accuracy_cart, precision_cart, recall_cart, f1_cart]

# Наївний Баєс
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
accuracy_nb, precision_nb, recall_nb, f1_nb, y_pred_nb = evaluate_model(model_nb, X_test, y_test)
model_results['NB'] = [accuracy_nb, precision_nb, recall_nb, f1_nb]

# SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)
accuracy_svm, precision_svm, recall_svm, f1_svm, y_pred_svm = evaluate_model(model_svm, X_test, y_test)
model_results['SVM'] = [accuracy_svm, precision_svm, recall_svm, f1_svm]

# Виведення результатів
for model_name in model_results:
    accuracy, precision, recall, f1 = model_results[model_name]
    print(f"\nМодель: {model_name}")
    print(f"Точність: {accuracy*100:.2f}%")
    print(f"Точність (Precision): {precision*100:.2f}%")
    print(f"Повнота (Recall): {recall*100:.2f}%")
    print(f"F1-міра: {f1*100:.2f}%")
