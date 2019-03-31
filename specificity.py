import pandas
from sklearn import model_selection
from numpy import nan
import numpy as np
from sklearn.metrics import classification_report,precision_score,recall_score,roc_auc_score,confusion_matrix
import sklearn.metrics as metric


def specif(yt,pa):
    cm = confusion_matrix(yt,pa)
    speci=0
    #print(cm.shape[0])
    for x in range(0,(cm.shape[0])):
        ts = np.sum(cm)
        tn = ts - np.sum(cm,axis=0)[x] - np.sum(cm,axis=1)[x]+cm[x,x]
        fp = np.sum(cm,axis=0)[x]-cm[x,x]
        if(tn==0 and fp==0):
            speci=speci+0
        else:
            speci=speci+tn/(tn+fp)
    return (speci/cm.shape[0])
