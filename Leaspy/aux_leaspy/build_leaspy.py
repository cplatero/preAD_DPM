import numpy as np
import pandas as pd
#import scipy.io
import pickle

from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters, Plotter#, __watermark__

from aux_leaspy.getMae import getMaeLeaspy
from aux_leaspy.getOnset import getOnset
from aux_leaspy.build_model_leaspy import buildModelLeaspy
from aux_leaspy.get_dps import getDPS
from aux_leaspy.leaspy_alg import leaspyPersonalize
from aux_leaspy.bayes_train import bayesTrain
from aux_leaspy.getAUC import getAUC

def buildLeaspy(df_train, df_test, percent, point_feat, id_train, id_test, diagnose_train, diagnose_test, 
                labels_train, labels_test, ages_train, ages_test, markers, ascending, btstrp,
                classes, classes_auc, save_model_path):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    build_leaspy.py
# Description: Building DPMs using Leaspy
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================  
    #DPM train
    Id_train, Idx_train, Tau_train, Xi_train, DPS_train = buildModelLeaspy(
        df_train, markers, btstrp, diagnose_train, classes, save_model_path)
    
    #Test personalize
    n_test = len(id_test)
    num_points_test = np.array([sum(df_test["ID"] == id) for id in id_test])
    num_points_train = np.array([sum(df_train["ID"] == id) for id in id_train])
    max_visits_test = num_points_test.max()
    Xi_test = np.zeros((btstrp, n_test), dtype=float)
    Tau_test = np.zeros((btstrp, n_test), dtype=float)
    DPS_test = np.full((btstrp, n_test, max_visits_test), np.nan) 
    MAE_test = np.full((btstrp, len(markers)), np.nan)
    Xi_train_all = np.full(Xi_train.shape, np.nan)
    Tau_train_all = np.full(Tau_train.shape, np.nan)
    DPS_train_all = np.full(DPS_train.shape, np.nan)
    MAE_train = np.full(MAE_test.shape, np.nan)
    
    for n in range(1, btstrp + 1):
        path = save_model_path + str(n) + ".json"
        model = Leaspy.load(path)
        _, ip_test_orig = leaspyPersonalize(df_test, markers, model)
        ip_test = pd.DataFrame(ip_test_orig._individual_parameters).T
        Tau_test[n-1, :] = ip_test.loc[id_test.astype(str)]["tau"]
        Xi_test[n-1, :] = ip_test.loc[id_test.astype(str)]["xi"]
        _, ip_train_orig = leaspyPersonalize(df_train, markers, model)
        ip_train = pd.DataFrame(ip_train_orig._individual_parameters).T
        Tau_train_all[n-1, :] = ip_train.loc[id_train.astype(str)]["tau"]
        Xi_train_all[n-1, :] = ip_train.loc[id_train.astype(str)]["xi"]
        
        #MAE
        MAE_test[n-1, :] = getMaeLeaspy(df_test, percent, point_feat, markers, model, ip_test_orig)
        MAE_train[n-1, :] = getMaeLeaspy(df_train, percent, point_feat, markers, model, ip_train_orig)
    
        for i, id in enumerate(id_test): 
            tij = df_test[df_test["ID"] == id]["TIME"]
            tau = Tau_test[n-1, i]
            xi = Xi_test[n-1, i]
            dps = getDPS(tij, tau, xi)
            DPS_test[n-1, i, : num_points_test[i]] = dps 
        
        for i, id in enumerate(id_train): 
            tij = df_train[df_train["ID"] == id]["TIME"]
            tau = Tau_train_all[n-1, i]
            xi = Xi_train_all[n-1, i]
            dps = getDPS(tij, tau, xi)
            DPS_train_all[n-1, i, : num_points_train[i]] = dps 
               
    #Bayesian classifier train
    posterior_bayes = np.empty(btstrp, dtype='O')
    likelihood = np.empty(btstrp, dtype='O')
    prior_bayes = np.empty(btstrp, dtype='O')
    evidence = np.empty(btstrp, dtype='O')

    for n in range(1, btstrp + 1):
        posterior_bayes[n-1], likelihood[n-1], prior_bayes[n-1], evidence[n-1] = bayesTrain(
            DPS_train[n-1, :, :], labels_train[Idx_train[n-1, :], :], classes)
    
    #AUC test
    AUC_test = np.zeros(btstrp, dtype = float)
    for n in range(1, btstrp + 1):
         AUC_test[n-1] = getAUC(DPS_test[n-1, :, :], posterior_bayes[n-1], 
                           likelihood[n-1], prior_bayes[n-1], labels_test, classes_auc)
         
    #AUC train
    AUC_train = np.zeros(btstrp, dtype = float)
    for n in range(1, btstrp + 1):
         AUC_train[n-1] = getAUC(DPS_train_all[n-1, :, :], posterior_bayes[n-1], 
                           likelihood[n-1], prior_bayes[n-1], labels_train,
                           classes_auc)
    
    #Metrics
    acc_scu_test, acc_smci_test, acc_pcu_test, corr_MCI_age_test, corr_MCI_reserve_test = \
        getOnset(DPS_test, posterior_bayes, prior_bayes, likelihood, ages_test, labels_test, btstrp)
    
    acc_scu_train, acc_smci_train, acc_pcu_train, corr_MCI_age_train, corr_MCI_reserve_train = \
        getOnset(DPS_train_all, posterior_bayes, prior_bayes, likelihood, ages_train, labels_train, btstrp)
    
    #Save data
    data = {"AUC_train" : AUC_train,
            "Id_train" : Id_train,
            "Idx_train" : Idx_train,
            "Tau_train" : Tau_train,
            "Xi_train" : Xi_train,
            "DPS_train" : DPS_train,
            "DPS_train_all" : DPS_train_all,
            "labels_train" : labels_train,
            "ages_train" : ages_train,
            "AUC_test" : AUC_test,
            "id_test" : id_test,
            "Tau_test" : Tau_test,
            "Xi_test" : Xi_test,
            "DPS_test" : DPS_test,
            "labels_test" : labels_test,
            "ages_test" : ages_test,
            "btstrp" : btstrp,
            "MAE_test" : MAE_test,
            "percent" : percent, 
            "point_feat" : point_feat,
            "df_train" : df_train,
            "df_test" : df_test
            }
    
    #scipy.io.savemat(save_model_path + "data.mat", data)
    
    with open(save_model_path + ".pkl", "wb") as file:
        pickle.dump(data, file)
         
    return AUC_test, AUC_train, acc_scu_test, acc_smci_test, acc_pcu_test, corr_MCI_age_test, \
            corr_MCI_reserve_test, acc_scu_train, acc_smci_train, acc_pcu_train, corr_MCI_age_train, \
            corr_MCI_reserve_train, MAE_test, MAE_train





