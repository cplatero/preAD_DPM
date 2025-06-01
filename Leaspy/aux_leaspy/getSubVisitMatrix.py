import numpy as np

def getSubVisitMatrix(df, col):
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getSubVisitMatrix.py
# Description: Dataframe to matrix for a given column
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   Leaspy
# ============================================================
    ids = df["ID"].unique()
    dx_per_sub = df.groupby("ID")[col]
    ncol = dx_per_sub.size().max()
    
    col = np.full((len(ids), ncol), np.nan, dtype='O')
    
    for idx, id in enumerate(ids):
        dx_sub = dx_per_sub.get_group(id)
        col[idx, :len(dx_sub)] = dx_sub
        
    return col
