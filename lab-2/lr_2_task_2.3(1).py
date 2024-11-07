# Імпортуємо необхідні бібліотеки
from sklearn.datasets import load_iris

# Завантажуємо набір даних Iris
iris_dataset = load_iris()

# Переглядаємо ключі словника
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

# Виводимо короткий опис набору даних
print(iris_dataset['DESCR'][:193] + "\n...")

print("Назви відповідей: {}".format(iris_dataset['target_names']))

print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))

print("Перші 5 рядків даних:\n{}".format(iris_dataset['data'][:5]))

print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Форма масиву target: {}".format(iris_dataset['target'].shape))

print("Відповіді:\n{}".format(iris_dataset['target']))

