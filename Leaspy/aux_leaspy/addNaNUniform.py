import numpy as np
import random

from aux_leaspy.getSubVisitMatrix import getSubVisitMatrix

def addNaNUniform(df, feats, perc_nan):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    addNaNUniform.py
# Description: Robustness to missing data
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    y = np.array([getSubVisitMatrix(df, str(feat)) for feat in feats]).astype(float)
    mask_nan_visits = np.isnan(getSubVisitMatrix(df, "TIME").astype(float))
    
    K, J, I = y.shape
    
    num_features = np.sum(~mask_nan_visits) * K
    
    nan_init = 0
    for k in range(0, K):
        feature_ = np.squeeze(y[k,:,:])
        feature_ = feature_[~mask_nan_visits]
        nan_init += np.sum(np.isnan(feature_))
        
    desired_nan = round(num_features * perc_nan / 100)
    nan_to_add = round((desired_nan - nan_init)/K)
    
    if nan_to_add > 0:
        for k in range(0, K):
            feature_ = np.squeeze(y[k,:,:])
            feature1D = np.ravel(feature_[~mask_nan_visits])
            no_nan_idx = np.where(~np.isnan(feature_[~mask_nan_visits]))[0]
            
            if k + 1 < K:
                feature_next = np.squeeze(y[k+1, :,:])
                feature1D_next = feature_next[~mask_nan_visits]
                nan_idx_next = np.where(np.isnan(feature1D_next))[0]
                no_nan_idx = np.array(list(set(no_nan_idx) - set(nan_idx_next)))
                
            if k + 1 > 1:
                feature_prev = np.squeeze(y[k-1,:,:])
                feature1D_prev = feature_prev[~mask_nan_visits]
                nan_idx_prev = np.where(np.isnan(feature1D_prev))[0]
                no_nan_idx = np.array(list(set(no_nan_idx) - set(nan_idx_prev)))
                
            idx_rand = random.sample(list(no_nan_idx), nan_to_add)
            
            feature1D[idx_rand] = np.nan
            feature_[~mask_nan_visits] = feature1D
            y[k,:,:] = feature_
            
    for k in range(0, K):
        df[feats[k]] = y[k,:,:][~mask_nan_visits]
                
    return df, y