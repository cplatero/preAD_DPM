import pandas as pd
import numpy as np

from scipy.stats import pearsonr

from aux_leaspy.getTrueOnset import getTrueOnset

def getOnset(DPS_test, posterior_bayes, prior_bayes, likelihood_bayes, 
             ages_test, labels_test, btstrp):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getOnset.py
# Description: Script for evaluating the Leaspy model (clinical scores)
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    I = DPS_test.shape[1] #Number of subjects
    prob_MCI = np.full(DPS_test.shape, np.nan)
    ages_onset = np.full((btstrp, I), np.nan)
    
    #MCI probability
    for n in range(0, btstrp):
        dps_test = DPS_test[n, :, :] 
        for i in range(0, dps_test.shape[0]):
            dps_subj = dps_test[i, :]
            mask_values = ~np.isnan(dps_subj)
            prob_MCI[n, i, mask_values] = posterior_bayes[n](
                likelihood_bayes[n], prior_bayes[n], dps_subj[mask_values])[1]
            
    for n in range(1, btstrp + 1):
        dps_test_n = DPS_test[n-1, :, :]
        prob_MCI = np.vectorize(lambda x : np.nan if np.isnan(x) else 1.0 if x > 0.5 else 0.0)(prob_MCI)
        
        for i in range(0, I):
            if np.nansum(prob_MCI[n-1,i, :] > 0):
                mask_values = ~np.vectorize(lambda x : x != x)(ages_test[i,:]) # Nan != Nan
                age_subj = ages_test[i, mask_values]
                label_subj = prob_MCI[n-1, i, mask_values]
                idx = np.where(label_subj == 1)[0]
                
                if label_subj[-1]:
                    if idx[0] > 1:
                        ages_onset[n-1, i] = (age_subj[idx[0]] + age_subj[idx[0]-1])/2
                    else:
                        ages_onset[n-1, i] = age_subj[idx[0]] 
    
    true_onset = getTrueOnset(labels_test, ages_test)
    
    #Metrics
    ages_bsl = ages_test[:, 0]
    
    acc_scu = np.full((btstrp, 1), np.nan)
    acc_smci = np.full((btstrp, 1), np.nan)
    acc_pcu = np.full((btstrp, 1), np.nan)
    corr_MCI_age = np.full((btstrp, 1), np.nan)
    corr_MCI_reserve = np.full((btstrp, 1), np.nan)
    
    n_scu = np.sum(np.isnan(true_onset))
    n_smci = np.sum(true_onset == ages_bsl)
    
    for n in range(0, btstrp):
        acc_scu[n] = 100 * np.sum(np.isnan(true_onset) & np.isnan(ages_onset[n, :])) / n_scu
        acc_smci[n] = 100 * np.sum((ages_onset[n, :] == ages_bsl) & (true_onset == ages_bsl)) / n_smci
        
    n_pcu = np.sum(true_onset > ages_bsl)
    
    for n in range(0,btstrp):
        acc_pcu[n] = 100 * np.sum((ages_onset[n, :] > ages_bsl) & (true_onset > ages_bsl)) / n_pcu
        corr_data_age = pd.DataFrame(np.column_stack((ages_onset[n, :], true_onset[0, :]))).dropna()
        corr_MCI_age[n] = corr_data_age.corr().iloc[0,1]
        corr_data_reserve = np.array(pd.DataFrame(np.column_stack((ages_onset[n, :] - ages_bsl, 
                                                          true_onset[0, :] - ages_bsl))).dropna())
        corr_MCI_reserve[n], _ = pearsonr(corr_data_reserve[:, 0], corr_data_reserve[:, 1])
        
    return acc_scu, acc_smci, acc_pcu, corr_MCI_age, corr_MCI_reserve
        


