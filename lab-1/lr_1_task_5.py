import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

# Завантажуємо дані
df = pd.read_csv('data_metrics.csv')
print(df.head())

# Додаємо прогнозовані мітки на основі порога 0.5
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())

# Визначаємо функції для обчислення TP, FN, FP, TN


def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


# Перевіряємо результати
print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

# Функція для обчислення значень матриці помилок


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

# Визначаємо власну функцію для матриці помилок


def Khrobust_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP],
                     [FN, TP]])


# Перевірка відповідності результатів
assert np.array_equal(Khrobust_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), 'Khrobust_confusion_matrix() is not correct for RF'

assert np.array_equal(Khrobust_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), 'Khrobust_confusion_matrix() is not correct for LR'

# Визначаємо власну функцію для accuracy_score


def Khrobust_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples correctly predicted
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)


# Перевірка accuracy_score
assert Khrobust_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values, df.predicted_RF.values), 'Khrobust_accuracy_score failed on RF'

assert Khrobust_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(
    df.actual_label.values, df.predicted_LR.values), 'Khrobust_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (Khrobust_accuracy_score(
    df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (Khrobust_accuracy_score(
    df.actual_label.values, df.predicted_LR.values)))

# Визначаємо власну функцію для recall_score


def Khrobust_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


# Перевірка recall_score
assert Khrobust_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(
    df.actual_label.values, df.predicted_RF.values), 'Khrobust_recall_score failed on RF'

assert Khrobust_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(
    df.actual_label.values, df.predicted_LR.values), 'Khrobust_recall_score failed on LR'

print('Recall RF: %.3f' % (Khrobust_recall_score(
    df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (Khrobust_recall_score(
    df.actual_label.values, df.predicted_LR.values)))

# Визначаємо власну функцію для precision_score


def Khrobust_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positive samples that are actually positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


# Перевірка precision_score
assert Khrobust_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'Khrobust_precision_score failed on RF'

assert Khrobust_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'Khrobust_precision_score failed on LR'

print('Precision RF: %.3f' % (Khrobust_precision_score(
    df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (Khrobust_precision_score(
    df.actual_label.values, df.predicted_LR.values)))

# Визначаємо власну функцію для f1_score


def Khrobust_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = Khrobust_recall_score(y_true, y_pred)
    precision = Khrobust_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


# Перевірка f1_score
assert Khrobust_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(
    df.actual_label.values, df.predicted_RF.values), 'Khrobust_f1_score failed on RF'

assert Khrobust_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(
    df.actual_label.values, df.predicted_LR.values), 'Khrobust_f1_score failed on LR'

print('F1 RF: %.3f' % (Khrobust_f1_score(
    df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (Khrobust_f1_score(
    df.actual_label.values, df.predicted_LR.values)))

# Порівняння результатів для різних порогів
print('\nScores with threshold = 0.5')
print('Accuracy RF: %.3f' % (Khrobust_accuracy_score(
    df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (Khrobust_recall_score(
    df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (Khrobust_precision_score(
    df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (Khrobust_f1_score(
    df.actual_label.values, df.predicted_RF.values)))

print('\nScores with threshold = 0.25')
df['predicted_RF_025'] = (df.model_RF >= 0.25).astype('int')
print('Accuracy RF: %.3f' % (Khrobust_accuracy_score(
    df.actual_label.values, df['predicted_RF_025'].values)))
print('Recall RF: %.3f' % (Khrobust_recall_score(
    df.actual_label.values, df['predicted_RF_025'].values)))
print('Precision RF: %.3f' % (Khrobust_precision_score(
    df.actual_label.values, df['predicted_RF_025'].values)))
print('F1 RF: %.3f' % (Khrobust_f1_score(
    df.actual_label.values, df['predicted_RF_025'].values)))

# Побудова ROC-кривих
fpr_RF, tpr_RF, thresholds_RF = roc_curve(
    df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(
    df.actual_label.values, df.model_LR.values)

# Графік ROC-кривих
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1], [0, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Обчислення AUC
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

# Графік ROC-кривих з AUC у легенді
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
