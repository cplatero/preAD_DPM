# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    percentile_interp.R
# Description: Script for transforming data to percentile scale
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

balanceSamples <- function(features, group) {
  gr_name <- na.omit(unique(group))
  num_samples <- array(0, dim=c(length(gr_name), 1))
  
  for (i in 1:length(gr_name))
    num_samples[i] <- sum(group == gr_name[i], na.rm = TRUE)
  
  max_samples <- max(num_samples)
  feat_extra <- list()
  
  for (i in seq_along(gr_name)) {
    if ((max_samples - num_samples[i]) > 0) {
      feat <- features[ifelse(is.na(group == gr_name[i]), FALSE, group == gr_name[i])]
      index <- sample(seq_len(num_samples[i]), size = max_samples - num_samples[i], replace = TRUE)
      feat_extra[[i]] <- feat[index]
    }
  }
  
  features <- c(features, unlist(feat_extra))
  
  return(features)
}

ECDFunc <- function(data) {
  sorted_data <- sort(data)
  sorted_data_unique <- unique(sorted_data)
  prob <- ecdf(data)(sorted_data_unique)
  
  p_min <- min(prob)
  prob[prob == p_min] <- 0
  
  return(list(sorted_data_unique, prob))
}


percentile_interp <- function(features, group, percent = NULL, point_feat = NULL) {
  num_features <- ncol(features)
  feat2per <- as.data.frame(matrix(0, nrow = nrow(features), ncol = num_features))
  
  if (!is.null(percent) & !is.null(point_feat)) {
    for (i in seq_len(num_features)) {
      interp_func <- approxfun(point_feat[[i]], percent[[i]], method = "linear", rule = 2)
      feat2per[, i] <- interp_func(features[, i])
    }
  } else {
    percent <- list()
    point_feat <- list()
    for (i in seq_len(num_features)) {
      mask_nan <- !is.na(features[, i])
      
      balanced_features <- balanceSamples(features[mask_nan, i], group[mask_nan,])
      ecdf_result <- ECDFunc(balanced_features)
      point_feat[[i]] <- ecdf_result[[1]]
      percent[[i]] <- ecdf_result[[2]]
      
      interp_func <- approxfun(point_feat[[i]], percent[[i]], method = "linear", rule = 1)
      feat2per[, i] <- interp_func(features[, i])
    }
  }
  
  return(list(feat2per = feat2per, percent = percent, point_feat = point_feat))
}

