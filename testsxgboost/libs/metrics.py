#Original source: https://github.com/miguelgfierro/codebase/blob/master/python/machine_learning/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score, precision_score, recall_score, f1_score


def classification_metrics_binary(y_true, y_pred):
    m_acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred)
    m_precision = precision_score(y_true, y_pred)
    m_recall = recall_score(y_true, y_pred)
    report = {'Accuracy':m_acc, 'Precision':m_precision, 'Recall':m_recall, 'F1':m_f1}
    return report


def classification_metrics_binary_prob(y_true, y_prob):
    m_auc = roc_auc_score(y_true, y_prob)
    report = {'AUC':m_auc}
    return report


def classification_metrics_multilabel(y_true, y_pred, labels):
    m_acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred, labels, average='weighted')
    m_precision = precision_score(y_true, y_pred, labels, average='weighted')
    m_recall = recall_score(y_true, y_pred, labels, average='weighted')
    report = {'Accuracy':m_acc, 'Precision':m_precision, 'Recall':m_recall, 'F1':m_f1}
    return report


def binarize_prediction(y, threshold=0.5):
    y_pred = np.where(y > threshold, 1, 0)
    return y_pred
