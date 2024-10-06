from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантажуємо дані
data = pd.read_csv('data_multivar_nb.txt', sep=' ', header=None)
data.columns = ['Feature1', 'Feature2', 'Label']

# Розділяємо на ознаки та мітки
X = data[['Feature1', 'Feature2']]
y = data['Label']

# Розділяємо на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Навчаємо SVM класифікатор
svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)

# Прогнозуємо та розраховуємо метрики для SVM
y_pred_svm = svm_classifier.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
y_scores_svm = svm_classifier.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_scores_svm)
auc_svm = roc_auc_score(y_test, y_scores_svm)

# Навчаємо наївний байєсівський класифікатор
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Прогнозуємо та розраховуємо метрики для Наївного Байєса
y_pred_nb = nb_classifier.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
acc_nb = accuracy_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
y_scores_nb = nb_classifier.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_scores_nb)
auc_nb = roc_auc_score(y_test, y_scores_nb)

# Виводимо метрики
print("SVM Classifier Metrics:")
print("Confusion Matrix:")
print(cm_svm)
print("Accuracy:", acc_svm)
print("Recall:", recall_svm)
print("Precision:", precision_svm)
print("F1 Score:", f1_svm)
print("AUC:", auc_svm)

print("\nNaive Bayes Classifier Metrics:")
print("Confusion Matrix:")
print(cm_nb)
print("Accuracy:", acc_nb)
print("Recall:", recall_nb)
print("Precision:", precision_nb)
print("F1 Score:", f1_nb)
print("AUC:", auc_nb)

# Будуємо ROC-криві
plt.figure()
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.2f)' % auc_svm)
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = %0.2f)' % auc_nb)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
