import pandas as pd
import numpy as np

from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters, Plotter#, __watermark__
from scipy.interpolate import interp1d

from aux_leaspy.percentile_interp import percentile_interp

# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    leaspy_alg.py
# Description: Script for training, personalizing and predicting
#              with Leaspy
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================

def leaspyFit(df_train, markers, save= True, save_path= "./data/leaspyFit.json"):
    #Convert ID column to string
    df_train["ID"] = df_train["ID"].astype(str)
    #Set index for train and test data
    df_train = df_train.set_index(['ID', 'TIME'], verify_integrity=True).sort_index()
    #Leaspy model parameters
    leaspy_model = 'logistic'
    algo_settings = AlgorithmSettings('mcmc_saem', 
                                      n_iter=3000, 
                                      seed=0,
                                      progress_bar=False)
    #Create data structures for leaspy code
    data_train = Data.from_dataframe(df_train[markers])
    #Fit the data
    leaspy = Leaspy(leaspy_model, source_dimension=int(np.floor(np.sqrt(len(markers)))), noise_model="gaussian_diagonal")
    leaspy.fit(data_train, algo_settings)
    #Save the model
    if(save):
        leaspy.save(save_path, indent= 2)
    #Return the model
    return leaspy
   
def leaspyPersonalize(df_test, markers, leaspy_model):
    #Convert ID column to string
    df_test["ID"] = df_test["ID"].astype(str)
    #Set index for train and test data
    df_test = df_test.set_index(['ID', 'TIME'], verify_integrity=True).sort_index()
    #Create data structures for leaspy code
    data_test = Data.from_dataframe(df_test[markers])
    #Personalize the data
    settings_personalization = AlgorithmSettings('scipy_minimize', progress_bar=False, use_jacobian=True)
    ip = leaspy_model.personalize(data_test, settings_personalization)
    #Convert ip to dataframe
    df_ip = ip.to_dataframe()
    #Resetea el índice
    df_test = df_test.reset_index()
    #Merge individual parameters to df_test
    df_test = df_test.merge(df_ip[["tau", "xi", "sources_0"]], on= "ID", how="left")
    #Leaspy estimated conversion time
    df_test["leaspy_estimation"] = df_test["AGE"] - df_test["tau"]
    #Return individual parameters
    return df_test, ip

def leaspyEstimate(df_test, percent, point_feat, markers, model, ip):
    ids = df_test["ID"].unique()
    estimation = np.full(df_test[markers].shape, np.nan)
    markers_orig = [m + "_orig" for m in markers]
    
    for id in ids: 
        pos = np.sum(~np.isnan(estimation[:, 0]))
        time_subj = df_test[df_test["ID"] == id]["TIME"].to_numpy()
        estimation[pos: pos + len(time_subj), :] = model.estimate({str(id):time_subj}, 
                                                       ip._individual_parameters)[str(id)]
        
    estimation_orig = np.zeros(estimation.shape)
       
    #Convert to original scale
    for j in range(estimation.shape[1]):
        interp_func = interp1d(percent[j], point_feat[j], kind='linear', fill_value='extrapolate')
        estimation_orig[:,j] = interp_func(estimation[:, j])
        
    return estimation, estimation_orig


