# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    grace_estimate.R
# Description: Script for predicting with GRACE
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

grace.estimate <- function(df, percent, point_feat, feats, model, gamma0, alpha0, alpha1){
  estimation <- array(NA, dim = dim(df[, feats]))
  estimation_orig <- array(NA, dim = dim(df[, feats]))
  ghat <- array(NA, dim = dim(df[, feats]))

  for (i in 1:length(feats)){
    mfit <- model$fits[[i]]$monotone
    Wfd <- mfit$Wfdobj
    
    if(!is.null(Wfd)){
      beta <- mfit$beta
      estimation[, i] <- beta[1] + beta[2] * eval.monfd(df$ct + gamma0, Wfd) + alpha0[, i] + alpha1[, i] * df$ct
    }
    else{
      coeff <- mfit$coefficients
      estimation[, i] <- coeff[1] + coeff[2] * (df$ct + gamma0) + alpha0[, i] + alpha1[, i] * df$ct
    }
    
    estimation <- pmax(pmin(estimation, 1), 0)
    interp_func <- approxfun(percent[[i]], point_feat[[i]], method = "linear", rule = 1)
    estimation_orig[, i] <- interp_func(estimation[, i])
  }
  
  return(list(estimation = estimation, estimation_orig = estimation_orig))
}