# Завантаження бібліотек
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Розмір датасету
print(dataset.shape)

print("---------------------------------")

# Перші 20 рядків
print(dataset.head(20))

print("---------------------------------")

# Статистичне резюме
print(dataset.describe())

print("---------------------------------")

# Розподіл за класом
print(dataset.groupby('class').size())

# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Гістограми
dataset.hist()
plt.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
plt.show()

# Перетворимо датасет у масив NumPy
array = dataset.values
# Виділяємо X та y
X = array[:,0:4]
y = array[:,4]

# Розділяємо дані на навчальний та тестовий набори
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінюємо моделі
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів')
plt.show()

# Навчаємо модель
model = SVC(gamma='auto')
model.fit(X_train, Y_train)

# Передбачення
predictions = model.predict(X_validation)

# Оцінка
print("Точність:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions))

# Нова квітка
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

# Передбачення для нової квітки
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)
print("Прогноз для нової квітки:", prediction[0])
