import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from aux_leaspy.percentile_interp import percentile_interp

# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getDataADNI.py
# Description: Preprocess ADNI file
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================

def daysFromBaseline(df):  
    df['D'] = 0
    for id in df['ID'].unique():
        subject = df[df['ID'] == id]
        baseline_date = pd.to_datetime(subject['EXAMDATE'].iloc[0], format="%Y/%m/%d")
        df.loc[df['ID'] == id, 'D'] = (pd.to_datetime(subject['EXAMDATE'], format="%Y/%m/%d") - baseline_date).dt.days
    
    return df

def conversionTimeDays(df):
    def computeTime(subject):
        dx_bl = subject['DX'].iloc[0]
        for ivisit in range(1, len(subject)):
            visit = subject.iloc[ivisit]
            if dx_bl != visit['DX']:
                return ((visit['D'] + subject.iloc[ivisit - 1]['D']) / 2) / 365
        return np.nan
    
    df['CONVERSION_TIME_DAYS'] = 0
    df['CONVERT'] = 0
    
    for id in df['ID'].unique():
        subject = df[df['ID'] == id]
        if len(subject) > 1 and (subject['DX'].iloc[0] != subject['DX'].iloc[-1]):
            df.loc[df['ID'] == id, 'CONVERSION_TIME_DAYS'] = computeTime(subject)
            df.loc[df['ID'] == id, 'CONVERT'] = 1
        else:
            df.loc[df['ID'] == id, 'CONVERT'] = 0
    
    return df


def centeredTimeDays(df):
    df['onset'] = 0
    for id in df['ID'].unique():
        subject = df[df['ID'] == id]
        dx_bl = subject['DX'].iloc[0]
        convert_bl = subject['CONVERT'].iloc[0]
        
        if dx_bl == "CN" and convert_bl == 1:  # pCU
            conversion_time_days = subject['CONVERSION_TIME_DAYS'].iloc[0] * 365
            df.loc[df['ID'] == id, 'onset'] = (df.loc[df['ID'] == id, 'D'] - conversion_time_days) / 30.4167
        
        elif dx_bl == "CN" and convert_bl == 0:  # sCU
            last_d = subject['D'].iloc[-1]
            df.loc[df['ID'] == id, 'onset'] = (df.loc[df['ID'] == id, 'D'] - last_d) / 30.4167
        
        elif dx_bl == "MCI" and convert_bl == 0:  # sMCI
            df.loc[df['ID'] == id, 'onset'] = df.loc[df['ID'] == id, 'D'] / 30.4167
    
    return df


def getDataADNI(path):
    df = pd.read_csv(path)
    df.rename(columns={"RID" : "ID"}, inplace = True)
    df = df.dropna(subset=["AGE", "DX"])
    
    ids_ad = df[(df["DX_bl"] == "AD") | ((df["VISCODE"] == "bl") & (df["DX"] == "Dementia"))]["ID"]
    df = df[~df["ID"].isin(ids_ad)]

    df["TIME"] = df["AGE"] + df["M"]/12

    df = daysFromBaseline(df)
    df = conversionTimeDays(df)
    df = centeredTimeDays(df)
    
    # cn --> 0, mci --> 1
    df["group"] = 1
    ids_cn = df.loc[(df["DX"] == "CN") & (df["M"] == 0), "ID"]
    df.loc[df["ID"].isin(ids_cn), "group"] = 0
    
    # scu, pcu, smci
    mask = (df["group"] == 0) | ((df["group"] == 1) & (df["CONVERT"] == 0)) 
    df = df[mask]
    
    mask_ad = df["DX"] == "Dementia"
    df = df[~mask_ad]
       
    n=2
    num_visits = df.groupby("ID").size().reset_index(name="num_visits")
    ids_few = num_visits[num_visits["num_visits"] < n]["ID"].tolist()
    
    df = df[~df["ID"].isin(ids_few)]
       
    return df

def prepareData(df, feats, asc):
    cols = ["ID", "TIME", "AGE", "DX", "EXAMDATE", "M", "D", "group", 
            "CONVERT", "CONVERSION_TIME_DAYS", "onset", "DX_bl"] +  feats
    
    for i, feat in enumerate(feats):
        if asc[i] == False:
            df[feat] = -df[feat]
            
    feats_orig = [feat + "_orig" for feat in feats]
    feats_dict = dict(zip(feats, feats_orig))
    df = df[cols]
    df.rename(columns = feats_dict, inplace = True)
    
    df["ID"] = df["ID"].astype(str)
    ids = df["ID"].unique()
    id_train, id_test = train_test_split(ids, test_size=0.2, random_state=42)
    id_train = np.sort(id_train.astype(int)).astype(str)
    idx_train = np.isin(ids, id_train)
    id_test = np.sort(id_test.astype(int)).astype(str)
    idx_test = np.isin(ids, id_test)
    
    df_train = df[df["ID"].isin(id_train)]
    df_test = df[df["ID"].isin(id_test)]
    
    group_train = pd.DataFrame([0 if (mci, conv) == (0,0) else 
             1 if mci == 1 and conv == 0 else 
             2 if mci == 0 and conv > 0 else None
                 for mci,conv in zip(df_train["group"], df_train["CONVERSION_TIME_DAYS"])])
    
    group_test = pd.DataFrame([0 if (mci, conv) == (0,0) else 
             1 if mci == 1 and conv == 0 else 
             2 if mci == 0 and conv > 0 else None
                 for mci,conv in zip(df_test["group"], df_test["CONVERSION_TIME_DAYS"])])

    feat2per_train, percent, point_feat = percentile_interp(
        (lambda df_train: 
             df_train[feats_orig].rename(columns=
                                  {col: i for i, col in enumerate(df_train[feats_orig].columns)}))(df_train), 
        group_train)
        
    feat2per_test, _, _ = percentile_interp(
        (lambda df_test: 
         df_test[feats_orig].rename(columns=
                              {col: i for i, col in enumerate(df_test[feats_orig].columns)}))(df_test), 
    group_test, percent, point_feat)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    df_train[feats] = feat2per_train
    df_test[feats] = feat2per_test
    
    return df_train, df_test, percent, point_feat, id_train, id_test, idx_train, idx_test
    
    
     
    
    
    



