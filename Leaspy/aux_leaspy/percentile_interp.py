import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    percentile_interp.py
# Description: Script for transforming data to percentile scale
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================

def balanceSamples(features, group):
    gr_name = np.unique(group)
    num_samples = np.zeros((len(gr_name), 1), dtype = int)
    
    for i in range(len(gr_name)):
        num_samples[i] = sum(group[0] == gr_name[i])
    
    max_samples = num_samples.max()
    feat_extra = np.array([])
    
    for i in range(len(gr_name)):
        if (max_samples - num_samples[i]) > 0 :
            feat = features[group[0] == gr_name[i]]
            index = np.random.randint(num_samples[i], size = max_samples - num_samples[i])
            feat_extra = np.append(feat_extra, [feat[i] for i in index])
            
    features = np.append(features, feat_extra)
    
    return features

def ecdf(data):
    sorted_data = np.sort(data)
    sorted_data_unique = np.unique(sorted_data)
    prob = ECDF(data)(sorted_data_unique)
    
    p_min = np.min(prob)
    prob = np.where(prob == p_min, 0, prob)
        
    return sorted_data_unique, prob

def percentile_interp(features, group, percent = None, point_feat = None):
    num_features = features.shape[1]
    feat2per = pd.DataFrame(np.zeros_like(features))
    
    if percent is not None and point_feat is not None:
        for i in range(num_features):
            interp_func = interp1d(point_feat[i], percent[i], kind='linear', fill_value='extrapolate')
            feat2per.iloc[:,i] = interp_func(features.iloc[:, i])
    else:
        percent = []
        point_feat = []
        for i in range(num_features):
            mask_nan = ~features.iloc[:, i].isna()

            feat, prob = ecdf(balanceSamples(np.array(features.loc[mask_nan, i]), 
                                             group[mask_nan.values]))
            point_feat.append(feat)
            percent.append(prob)
            
            interp_func = interp1d(point_feat[i], percent[i], kind='linear', fill_value='extrapolate')
            feat2per.iloc[:,i] = interp_func(features.iloc[:, i])
    
    return feat2per, percent, point_feat



