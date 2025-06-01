import numpy as np

def getDiagnose(df):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    get_diagnose.py
# Description: First and last diagnoses
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    ids = df["ID"].unique()
    diagnose = np.zeros((2, len(ids)), dtype= 'O')
    
    for idx, id in enumerate(ids):
        df_sub = df[df["ID"] == id]
        df_sub_dx = df_sub["DX"]
        diagnose[0,idx] = df_sub_dx.iloc[0]
        diagnose[1,idx] = df_sub_dx.iloc[-1]
        
    return diagnose