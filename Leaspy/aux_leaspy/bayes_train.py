import numpy as np

from scipy.stats import gaussian_kde

def bayesTrain(dps_train, labels, unique_labels):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    bayes_train.py
# Description: Bayes classifier train
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================    
    def evidence (likelihood, prior, points):
        ev = np.zeros_like(points).astype(np.float64)
        for i, (_,model) in enumerate(likelihood.items()):
            ev += model(points) * prior[i]
        return ev

    def posterior(likelihood, prior, points):
        ev = evidence(likelihood, prior, points)
        post = dict()
        
        for i, (_,model) in enumerate(likelihood.items()):
            post[i] = (prior[i] * likelihood[i](points)) / ev
            
        return post
   
    labels = labels.ravel()
    mask = np.vectorize(lambda x : x != x)(labels) # Nan != Nan
    labels = labels[~mask]
    dps_train = dps_train.ravel()
    dps_train = dps_train[~np.isnan(dps_train)]
    
    
    
    C = len(unique_labels)
    prior = np.zeros(C)
    likelihood = {}

    for c in range(C):
        prior[c] = np.sum(labels == unique_labels[c]) / len(labels)
        model = gaussian_kde(dps_train[labels == unique_labels[c]].astype(float), bw_method='scott') #KDE model
        likelihood[c] = model #Probability in point x is likelihood[c](x)

    return posterior, likelihood, prior, evidence