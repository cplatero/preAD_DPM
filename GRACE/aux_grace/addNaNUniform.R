addNaNUniform <- function(df, feats, perc_nan){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    addNaNUniform.R
  # Description: Robustness to missing data
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  K <- length(feats)
  ages <- getSubVisitMatrix(df, "TIME")
  I <- dim(ages)[1]
  J <- dim(ages)[2]
  
  mask_nan_visits <- t(is.na(ages))
  
  y <- array(NA, dim = c(K, J, I))
  
  for (k in 1:K)
    y[k,,] <- t(getSubVisitMatrix(df, feats[k]))
  
  num_features <- sum(!mask_nan_visits) * K
  
  nan_init <- 0
  for (k in 1:K) {
    feature_ <- y[k, , ]
    feature_ <- feature_[!mask_nan_visits]
    nan_init <- nan_init + sum(is.na(feature_))
  }
  
  desired_nan <- round(num_features * perc_nan / 100)
  nan_to_add <- round((desired_nan - nan_init) / K)
  
  if (nan_to_add > 0) {
    for (k in 1:K) {
      feature_ <- y[k, , ]
      feature1D <- as.vector(feature_[!mask_nan_visits])
      no_nan_idx <- which(!is.na(feature1D))
      
      if (k < K) {
        feature_next <- y[k + 1, , ]
        feature1D_next <- as.vector(feature_next[!mask_nan_visits])
        nan_idx_next <- which(is.na(feature1D_next))
        no_nan_idx <- setdiff(no_nan_idx, nan_idx_next)
      }
      
      if (k > 1) {
        feature_prev <- y[k - 1, , ]
        feature1D_prev <- as.vector(feature_prev[!mask_nan_visits])
        nan_idx_prev <- which(is.na(feature1D_prev))
        no_nan_idx <- setdiff(no_nan_idx, nan_idx_prev)
      }
      
      idx_rand <- sample(no_nan_idx, nan_to_add)
      
      feature1D[idx_rand] <- NA
      feature_[!mask_nan_visits] <- feature1D
      y[k, , ] <- feature_
    }
  }

  for (k in 1:K) {
    df[[feats[k]]] <- y[k, , ][!mask_nan_visits]
  }
  
  return(list(df = df, y = y))
  
}