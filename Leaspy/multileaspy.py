import os
path_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(path_dir)


import pandas as pd
import numpy as np

from itertools import combinations
# from sklearn.model_selection import train_test_split

from aux_leaspy.get_diagnose import getDiagnose
from aux_leaspy.getSubVisitMatrix import getSubVisitMatrix
from aux_leaspy.build_leaspy import buildLeaspy
from aux_leaspy.getDataADNI import  prepareData
from aux_leaspy.addNaNUniform import addNaNUniform


def multiLeaspy():
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    multiLeaspy.py
# Description: Script for training the Leaspy model
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    
    # adni_file = "./data/ADNIMERGE_220706.csv"
    header = "./data_multi/preclinical_CAM_jul22_"
    vector_range = range(2, 2 + 1)
    features_set = [5, 8]
    btstrp = 2
    percent_rem = 0
    
    classes = ["CN", "MCI"]
    classes_auc = ["CN", "MCI"]
       
    markers_df = pd.DataFrame({
    'markers': ["RAVLT_forgetting", "RAVLT_immediate", "RAVLT_learning", 
                "RAVLT_perc_forgetting", "ADAS13", "FAQ", "MMSE", "CDRSB", 
                "LDELTOTAL", "mPACCdigit", "mPACCtrailsB", "TRABSCOR"],
    'ascending': [True, False, False, 
                  True, True, True, False, True, 
                  False, False, False, True],
    'id': [1, 2, 3, 
           4, 5, 6, 7, 8, 
           9, 10, 11, 12]
    })
    
    feat_sub = pd.DataFrame(columns = [
                                        [f"feat_{i}" for i in range(1,max(vector_range)+1)] + 
                                        [f"id_{i}" for i in range(1,max(vector_range)+1)] +
                                        [f"ascending_{i}" for i in range(1,max(vector_range)+1)] +
                                        ["dir"]
                                    ])
    
    feats = list(markers_df["markers"])
    asc = list(markers_df["ascending"])
    feats_selected = list(markers_df[markers_df["id"].isin(features_set)]["markers"])
    
    #Results table     
    results_table = pd.DataFrame()
    
    columns = [f"feat_{i}" for i in range(1, max(vector_range) +1)] + \
                [f"id_{i}" for i in range(1, max(vector_range) +1)] + \
                ["Corr_MCI_age_test", "Corr_MCI_reserve_test", "acc_scu_test",
                 "acc_smci_test", "acc_pcu_test", "AUC_test", 
                 "Corr_MCI_age_train", "Corr_MCI_reserve_train", "acc_scu_train", 
                 "acc_smci_train", "acc_pcu_train", "AUC_train", 
                 "std_corr_MCI_age_test", "std_corr_MCI_reserve_test", "std_acc_scu_test",
                 "std_acc_smci_test", "std_acc_pcu_test", "std_AUC_test", 
                 "std_corr_MCI_age_train", "std_corr_MCI_reserve_train", "std_acc_scu_train",
                 "std_acc_smci_train", "std_acc_pcu_train", "std_AUC_train"]
    
    for col in columns:
        results_table[col] = []
    
    # MAE table
    MAE_table_test = pd.DataFrame()
    MAE_table_train = pd.DataFrame()
    
    MAE_columns = ["feat_id"] + \
              [m for m in markers_df[markers_df["id"].isin(features_set)]["markers"]] + \
              ["std_" + str(m) for m in markers_df[markers_df["id"].isin(features_set)]["markers"]]
        
    for col in MAE_columns:
        MAE_table_test[col] = []
        MAE_table_train[col] = []
               
    #Train-test split
    #df = getDataADNI(adni_file)
    df=pd.read_csv("./data/df.csv")    
    df, _ = addNaNUniform(df, feats_selected, percent_rem)
    df_train, df_test, percent, point_feat, _, id_test, _, _ = prepareData(df, feats, asc)
    
    # Create table with feats and directories to save data
    for n in vector_range:
        ncomb = list(combinations(markers_df.loc[markers_df["id"].isin(features_set), "markers"],n))
        
        for comb in ncomb:
            feat = [item for item in comb]
            id = list(map(lambda x: markers_df[markers_df["markers"] == x]["id"].iloc[0],feat))
            ascending = list(map(lambda x: markers_df[markers_df["markers"] == x]["ascending"].iloc[0],feat))
            
            dir = [header + "_".join(map(str, id))]
            feat = feat + [np.nan] * (max(vector_range) - n)
            id = id + [np.nan] * (max(vector_range) - n)
            ascending = ascending + [np.nan] * (max(vector_range) - n)
            
            row = feat + id + ascending + dir

            feat_sub = feat_sub.append(pd.Series(row, index = feat_sub.columns), ignore_index=True)
    
    for idx, row in feat_sub.iterrows():
        dir = row["dir"]
        file = dir + "/dd.xlsx"
        
        if os.path.exists(file):
            continue
        
        else:
            if not os.path.exists(dir):
                os.makedirs(dir)
                
            feat = row.filter(regex="feat*")
            feat = list(feat.dropna().values)
            feat_orig = [f + "_orig" for f in feat]
            
            cols = ["ID", "AGE", "TIME", "EXAMDATE", "M", "DX", "D", 
                    "CONVERSION_TIME_DAYS", "CONVERT", "onset", "group", "DX_bl"] + feat + feat_orig
            
            df_subs = df_train[cols]
            all_markers_nan_idx = df_subs[feat_orig].isna().all(axis=1)
            df_subs = df_subs[~all_markers_nan_idx]
            df_subs.to_excel(file, index = False)
        
    for idx, row in feat_sub.iterrows():
        dir = row["dir"]
        file_in = dir + "/dd.xlsx"
        save_model_path = dir + "/leaspy_btstrp"
        
        if os.path.exists(save_model_path + "_data.pkl"):
            continue
        
        else:
            feat = row.filter(regex="feat*")
            feat = list(feat.dropna().values)
            asc = row.filter(regex="ascending*")
            asc = list(asc.dropna().values)
            ids = row.filter(regex="id_")
            ids = list(ids.dropna().values)
            dd = pd.read_excel(file_in)
            dd["ID"] = dd["ID"].astype(str)
            percent_sub = [percent[i-1] for i in ids]
            point_feat_sub = [point_feat[i-1] for i in ids]
            id_train = dd["ID"].unique()
            #Clean test data
            all_markers_nan_idx = df_test[feat].isna().all(axis=1)
            df_test_subs = df_test[~all_markers_nan_idx]
            id_test_subs = df_test_subs["ID"].unique()
            #Diagnose
            diagnose_train = getDiagnose(dd)
            diagnose_test = getDiagnose(df_test_subs)
            #Labels
            labels_train = getSubVisitMatrix(dd, "DX")
            labels_test = getSubVisitMatrix(df_test_subs, "DX")
            #Ages
            ages_train = getSubVisitMatrix(dd, "TIME")
            ages_test = getSubVisitMatrix(df_test_subs, "TIME")
            
            AUC_test, AUC_train, acc_scu_test, acc_smci_test, acc_pcu_test, corr_MCI_age_test, \
            corr_MCI_reserve_test, acc_scu_train, acc_smci_train, acc_pcu_train, corr_MCI_age_train, \
            corr_MCI_reserve_train, MAE_test, MAE_train = \
            buildLeaspy(dd, df_test_subs, percent_sub, point_feat_sub, id_train, id_test_subs, 
                        diagnose_train, diagnose_test, labels_train,
                        labels_test, ages_train, ages_test, feat, asc, btstrp, 
                        classes, classes_auc, save_model_path)
            
            AUC_test_mean = np.mean(AUC_test)
            AUC_test_std = np.std(AUC_test)
            AUC_train_mean = np.mean(AUC_train)
            AUC_train_std = np.std(AUC_train)
            acc_scu_test_mean = np.mean(acc_scu_test)
            acc_scu_test_std = np.std(acc_scu_test)
            acc_smci_test_mean = np.mean(acc_smci_test)
            acc_smci_test_std = np.std(acc_smci_test)
            acc_pcu_test_mean = np.mean(acc_pcu_test)
            acc_pcu_test_std = np.std(acc_pcu_test)
            corr_MCI_age_test_mean = np.mean(corr_MCI_age_test)
            corr_MCI_age_test_std = np.std(corr_MCI_age_test)
            corr_MCI_reserve_test_mean = np.mean(corr_MCI_reserve_test)
            corr_MCI_reserve_test_std = np.std(corr_MCI_reserve_test)
            acc_scu_train_mean = np.mean(acc_scu_train)
            acc_scu_train_std = np.std(acc_scu_train)
            acc_smci_train_mean = np.mean(acc_smci_train)
            acc_smci_train_std = np.std(acc_smci_train)
            acc_pcu_train_mean = np.mean(acc_pcu_train)
            acc_pcu_train_std = np.std(acc_pcu_train)
            corr_MCI_age_train_mean = np.mean(corr_MCI_age_train)
            corr_MCI_age_train_std = np.std(corr_MCI_age_train)
            corr_MCI_reserve_train_mean = np.mean(corr_MCI_reserve_train)
            corr_MCI_reserve_train_std = np.std(corr_MCI_reserve_train)
            MAE_test_mean = MAE_test.mean(axis=0)
            MAE_test_std = MAE_test.std(axis=0)
            MAE_train_mean = MAE_train.mean(axis=0)
            MAE_train_std = MAE_train.std(axis=0)
            
            res = [feat[i] if (len(feat) > i) else "" for i in range(0,max(vector_range))] + \
                  [ids[i] if (len(feat) > i) else "" for i in range(0,max(vector_range))] + \
                  [corr_MCI_age_test_mean, corr_MCI_reserve_test_mean, 
                   acc_scu_test_mean, acc_smci_test_mean, acc_pcu_test_mean,
                   AUC_test_mean, corr_MCI_age_train_mean, corr_MCI_reserve_train_mean,
                   acc_scu_train_mean, acc_smci_train_mean, acc_pcu_train_mean,
                   AUC_train_mean,
                   corr_MCI_age_test_std, corr_MCI_reserve_test_std,
                   acc_scu_test_std, acc_smci_test_std, acc_pcu_test_std,
                   AUC_test_std, corr_MCI_age_train_std, corr_MCI_reserve_train_std,
                   acc_scu_train_std, acc_smci_train_std, acc_pcu_train_std,
                   AUC_train_std]
                
            res = dict(zip(results_table.columns, res))
            results_table = results_table.append(res, ignore_index=True)
            
            str_id = "_".join(map(str, ids))
            MAE_row_test = [str_id] + \
                      ["" if id not in ids else MAE_test_mean[ids.index(id)] for id in features_set] + \
                      ["" if id not in ids else MAE_test_std[ids.index(id)] for id in features_set]
                      
            MAE_row_train = [str_id] + \
                      ["" if id not in ids else MAE_train_mean[ids.index(id)] for id in features_set] + \
                      ["" if id not in ids else MAE_train_std[ids.index(id)] for id in features_set]
            
            MAE_row_test = dict(zip(MAE_table_test.columns, MAE_row_test))
            MAE_row_train = dict(zip(MAE_table_train.columns, MAE_row_train))
            MAE_table_test = MAE_table_test.append(MAE_row_test, ignore_index = True)
            MAE_table_train = MAE_table_train.append(MAE_row_train, ignore_index = True)
            
    results_xlsx = "preclinical_CAM_jul22_" + "_".join(map(str, markers_df["id"])) + ".xlsx"
    results_table.to_excel(results_xlsx, index = False)
    
    MAE_table_test.to_excel("MAE_test_preclinical_CAM_jul22.xlsx", index=False)
    MAE_table_train.to_excel("MAE_train_preclinical_CAM_jul22.xlsx", index=False)           
    
             
multiLeaspy()
