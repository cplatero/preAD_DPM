import numpy as np

def getTrueOnset(labels_test, ages_test):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getTrueOnset.py
# Description: Script for calculating conversion time
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    I = labels_test.shape[0]
    true_onset = np.full((1, I), np.nan)
    MCI = np.where(labels_test == "MCI", 1, 0)
    
    for i in range(0, I):
        if np.nansum(MCI[i,:] > 0):
            mask_values = ~np.vectorize(lambda x : x != x)(ages_test[i,:]) # Nan != Nan
            age_subj = ages_test[i, mask_values]
            label_subj = MCI[i, mask_values]
            idx = np.where(label_subj == 1)[0]
            
            if label_subj[-1]:
                if idx[0] > 1:
                    true_onset[0, i] = (age_subj[idx[0]] + age_subj[idx[0]-1])/2
                else:
                    true_onset[0, i] = age_subj[idx[0]] 
    
    return true_onset