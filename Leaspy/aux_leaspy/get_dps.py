import math

def getDPS(tij, tau, xi):
    # ============================================================
    # Project:    Disease progression modeling from early AD stage
    # Repository: https://github.com/cplatero/preAD_DPM
    # Author:     Jorge Bengoa
    # Email:      j.bpinedo@alumnos.upm.es
    # Institution:Universidad Politécnica de Madrid
    # ------------------------------------------------------------
    # Filename:    get_dps.py
    # Description: Leaspy DPS
    #
    # Version:    1.0
    # Date:       2025-05-09
    # Requires:   Leaspy
    # ============================================================
    return math.exp(xi)*(tij-tau)
