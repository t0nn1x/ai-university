import pandas as pd
import numpy as np

# Створюємо датафрейм з даними
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)


def calculate_probability(df, outlook, humidity, wind):
    # Розрахунок P(Yes) і P(No)
    total = len(df)
    p_yes = len(df[df['Play'] == 'Yes']) / total
    p_no = len(df[df['Play'] == 'No']) / total

    # Розрахунок умовних ймовірностей для "Yes"
    p_outlook_yes = len(df[(df['Outlook'] == outlook) & (
        df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])
    p_humidity_yes = len(df[(df['Humidity'] == humidity) & (
        df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])
    p_wind_yes = len(df[(df['Wind'] == wind) & (
        df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])

    # Розрахунок умовних ймовірностей для "No"
    p_outlook_no = len(df[(df['Outlook'] == outlook) & (
        df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])
    p_humidity_no = len(df[(df['Humidity'] == humidity) & (
        df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])
    p_wind_no = len(df[(df['Wind'] == wind) & (
        df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])

    # Розрахунок фінальних ймовірностей
    p_yes_final = p_outlook_yes * p_humidity_yes * p_wind_yes * p_yes
    p_no_final = p_outlook_no * p_humidity_no * p_wind_no * p_no

    # Нормалізація
    sum_p = p_yes_final + p_no_final
    p_yes_normalized = p_yes_final / sum_p
    p_no_normalized = p_no_final / sum_p

    return p_yes_normalized, p_no_normalized


# Розрахунок для варіанту 15
p_yes, p_no = calculate_probability(df, 'Rain', 'High', 'Strong')

print(f"Ймовірність 'Yes': {p_yes:.4f}")
print(f"Ймовірність 'No': {p_no:.4f}")
