import os
import pandas as pd
import numpy as np

from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters, Plotter#, __watermark__

from aux_leaspy.leaspy_alg import leaspyFit, leaspyPersonalize
from aux_leaspy.get_dps import getDPS


def buildModelLeaspy(df_train, markers, btstrp, diagnose, classes, save_model_path):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    build_model_leaspy.py
# Description: Building DPMs using Leaspy
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================  
    ids = df_train["ID"].unique()
    n_train = len(ids)
    num_points = np.array([sum(df_train["ID"] == id) for id in ids])
    max_visits = num_points.max()
    Idx_train = np.zeros((btstrp, n_train), dtype=int)
    Id_train = np.zeros((btstrp, n_train), dtype=int)
    Xi = np.zeros((btstrp, n_train), dtype=float)
    Tau = np.zeros((btstrp, n_train), dtype=float)
    DPS_Train = np.full((btstrp, n_train, max_visits), np.nan) 
    
    for n in range(1, btstrp + 1):
        idx_btstrp = []
        np.random.seed(n)
        
        for u in range(len(classes)):
            for uu in range(len(classes)):
                idx_u = np.where((diagnose[0] == classes[u]) & (diagnose[1] == classes[uu]))[0]
                med_u = np.median(num_points[idx_u])
                idx_many = idx_u[num_points[idx_u] > med_u]
                
                if len(idx_many) > 0:
                    idx_rnd = np.random.choice(idx_many, size=len(idx_many), replace=True)
                    idx_btstrp.extend(idx_rnd)
                    
                idx_few = idx_u[num_points[idx_u] <= med_u]
                
                if len(idx_few) > 0:
                    idx_rnd = np.random.choice(idx_few, size=len(idx_few), replace=True)
                    idx_btstrp.extend(idx_rnd)
        
        Idx_train[n-1, :] = idx_btstrp
        Id_train[n-1, :] = ids[idx_btstrp]
        
        #Train
        path = save_model_path + str(n) + ".json"
        df_train_btstrp = df_train[df_train["ID"].isin(Id_train[n-1,:].astype(str))]
        
        if not os.path.exists(path):
            model = leaspyFit(df_train_btstrp, markers, save=True, save_path=path)
            Leaspy.save(model, path)
        else:
            model = Leaspy.load(path)
        
        #Personalize
        _, ip = leaspyPersonalize(df_train_btstrp, markers, model)
        ip = pd.DataFrame(ip._individual_parameters).T
        Tau[n-1, :] = ip.loc[Id_train[n-1, :].astype(str)]["tau"]
        Xi[n-1, :] = ip.loc[Id_train[n-1, :].astype(str)]["xi"]
        
        #DPS train
        for i, (idx, id) in enumerate(zip(Idx_train[n-1, :], Id_train[n-1, :])): 
            tij = df_train[df_train["ID"] == id.astype(str)]["TIME"]
            tau = Tau[n-1, i]
            xi = Xi[n-1, i]
            dps = getDPS(tij, tau, xi)
            DPS_Train[n-1, i, : num_points[idx]] = dps
        
    return Id_train, Idx_train, Tau, Xi, DPS_Train


