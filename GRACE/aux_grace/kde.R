# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    kde.R
# Description: Script for Kernel Density Estimation of DPS
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

densityKDE <- function(dps_train, labels, unique_labels) {
  require(ks)
  labels <- unlist(labels)
  # labels <- as.vector(labels)
  mask <- is.na(labels)
  labels <- labels[!mask]
  dps_train <- unlist(dps_train)
  # dps_train <- as.vector(dps_train)
  dps_train <- dps_train[!is.na(dps_train)]
  
  C <- length(unique_labels)
  densities <- list()
  
  for (c in seq(1,C)){
    dps_train_c <- dps_train[labels == unique_labels[c]]
    densities[[c]] <- kde(dps_train_c)
  }
  
  return (densities)
}