from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

PROMOTION_RATE = 0.25
PROBA_CONVERT_PROMOTION = 0.25

from src import preprocessing,visualization

def print_scores(y,yh):
    print("F1 score : {}".format(f1_score(y,yh)))
    print("Accuracy score : {}".format(accuracy_score(y,yh)))
    print("Recall score : {}".format(recall_score(y,yh)))
    print("Precision score : {}".format(precision_score(y,yh)))

def compute_confusion_matrix(y,yh):
    cm = confusion_matrix(y,yh)
    return cm

# Wrapper for package consistency

def roc_aux_score(y,proba_churn):
    return roc_auc_score(y,proba_churn)
def compute_rates(y,proba_churn):
    fpr,tpr,_ = roc_curve(y,proba_churn)
    return fpr, tpr