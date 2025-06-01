import numpy as np

from aux_leaspy.leaspy_alg import leaspyEstimate

def getMaeLeaspy(df_test, percent, point_feat, markers, model, ip):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getMae.py
# Description: Script for evaluating the Leaspy model (MAE)
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    markers_orig = [m + "_orig" for m in markers]
    y_test = df_test[markers_orig]
    MAE = np.full(len(markers), np.nan)
    
    y_test_fit, y_test_fit_orig = leaspyEstimate(df_test, percent, point_feat, markers, model, ip)
    
    for i, m in enumerate(markers_orig): 
        MAE[i] = np.nanmean(abs(y_test_fit_orig[:, i] - y_test[m]))
        
    return MAE
        
