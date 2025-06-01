import numpy as np
from sklearn.metrics import roc_auc_score

def getAUC(dps_test, posterior_bayes, likelihood, prior, labels, classes):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getAUC.py
# Description: Script for evaluating the Leaspy model (AUC)
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    dps_test = dps_test.ravel()
    mask = ~np.isnan(dps_test)
    dps_test = dps_test[mask]
    
    #One vs one AUC
    labels = labels.ravel()
    mask = np.vectorize(lambda x : x != x)(labels) # Nan != Nan
    labels = labels[~mask]
    mask_class = np.isin(labels, classes)
    
    labels = labels[mask_class]
    dps_test = dps_test[mask_class]
    
    predict_scores = posterior_bayes(likelihood, prior, dps_test)
    predict_scores = np.column_stack(list(predict_scores.values()))
    
    #Convert labels to numeric
    mapping = {nlabel: i for i, nlabel in enumerate(classes)}
    labels = np.array([mapping[nlabel] for nlabel in labels])
    
    #Remove outliers 
    mask_out = np.isnan(predict_scores[:,0]) | np.isnan(labels)
    labels = labels[~mask_out]
    predict_scores = predict_scores[~mask_out]
    
    if len(classes) == 2:
        #Binary case: the score corresponds to the probability of the class 
        #with the greater label (i.e. 2 if labels are 1 and 2)
        labels = -labels
        auc = roc_auc_score(
            labels,
            predict_scores[:, 0],
            average="macro"
        )
    else:
        #Multiclass case: the order of the class scores corresponds to the 
        #order of the labels, numerically (i.e. 1,2,3 of labels are 1,2,3)
        auc = roc_auc_score(
            labels,
            predict_scores,
            multi_class="ovo", 
            average="macro",
        )
    return auc
