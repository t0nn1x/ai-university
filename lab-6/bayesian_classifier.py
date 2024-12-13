import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(url):
    # Завантаження датасету
    df = pd.read_csv(url)

    # Видалення рядків з відсутніми значеннями
    df = df.dropna(subset=['price', 'train_class', 'fare', 'train_type'])

    # Базова інформація про датасет
    print("\nІнформація про датасет після очистки:")
    print(df.info())

    # Статистичний опис числових даних
    print("\nСтатистичний опис:")
    print(df.describe())

    return df


def prepare_features(df):
    # Створення копії датафрейму
    df_prepared = df.copy()

    # Кодування категоріальних змінних
    le = LabelEncoder()
    categorical_columns = ['train_type', 'train_class', 'fare']

    for col in categorical_columns:
        df_prepared[col] = le.fit_transform(df_prepared[col])

    # Створення бінів для ціни
    df_prepared['price_category'] = pd.qcut(
        df_prepared['price'], q=5, labels=[0, 1, 2, 3, 4])

    return df_prepared


def train_naive_bayes(X, y):
    # Розділення даних на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Створення та навчання моделі
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Прогнозування
    y_pred = model.predict(X_test)

    # Оцінка точності
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочність моделі: {accuracy:.2f}")

    # Детальний звіт про класифікацію
    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred


def visualize_results(df, y_true, y_pred):
    plt.figure(figsize=(15, 5))

    # Графік розподілу цін
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='price', bins=30)
    plt.title('Розподіл цін на квитки')
    plt.xlabel('Ціна')
    plt.ylabel('Кількість')

    # Графік розподілу за типом потяга
    plt.subplot(1, 3, 2)
    df['train_type'].value_counts().plot(kind='bar')
    plt.title('Розподіл за типом потяга')
    plt.xlabel('Тип потяга')
    plt.ylabel('Кількість')

    # Матриця помилок
    plt.subplot(1, 3, 3)
    conf_matrix = pd.crosstab(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Матриця помилок')
    plt.xlabel('Прогнозовані значення')
    plt.ylabel('Реальні значення')

    plt.tight_layout()
    plt.savefig('train_analysis_results.png')
    plt.close()


def analyze_price_distribution(df):
    print("\nАналіз розподілу цін:")
    print("\nОсновні статистичні показники цін:")
    print(df['price'].describe())

    print("\nРозподіл за типами потягів:")
    print(df.groupby('train_type')['price'].agg(['mean', 'min', 'max']))


def main():
    # URL датасету
    url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"

    # Завантаження та підготовка даних
    df = load_and_prepare_data(url)

    # Аналіз розподілу цін
    analyze_price_distribution(df)

    # Підготовка ознак
    df_prepared = prepare_features(df)

    # Визначення ознак та цільової змінної
    X = df_prepared[['train_type', 'train_class', 'fare']]
    y = df_prepared['price_category']

    # Навчання моделі та отримання результатів
    model, X_test, y_test, y_pred = train_naive_bayes(X, y)

    # Візуалізація результатів
    visualize_results(df, y_test, y_pred)

    print("\nРезультати збережено у файл 'train_analysis_results.png'")


if __name__ == "__main__":
    main()
