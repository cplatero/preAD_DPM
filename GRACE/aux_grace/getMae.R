source("aux_grace/percentile_interp.R")

getMaeGrace <- function(dd, percent, point_feat, markers){
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getMae.R
# Description: Script for evaluating the GRACE model (MAE)
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================
  MAE <- array(NA, dim = c(length(markers)))
  markers_orig <- paste0(markers, "_orig")
  
  yi_hat_norm <- with(dd, ghat + alpha0 + alpha1 * t)
  yi_hat_norm <- ifelse(yi_hat_norm > 1, 1, 
                        ifelse(yi_hat_norm < 0, 0, yi_hat_norm))
  dd$yi_hat_norm <- yi_hat_norm
  list_yi_hat_norm <- split(dd$yi_hat_norm, dd$Outcome)
  list_y_norm <- split(dd$y, dd$Outcome)
  
  for (i in 1:length(markers)){
    interp_func <- approxfun(percent[[i]], point_feat[[i]], method = "linear", rule = 1)
    yi_hat <- interp_func(list_yi_hat_norm[[i]])
    y <- interp_func(list_y_norm[[i]])
    MAE[i] <- mean(abs(y - yi_hat), na.rm = TRUE)
  }
  
  return (MAE)
}

