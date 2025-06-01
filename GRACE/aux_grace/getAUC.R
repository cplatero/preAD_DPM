getAUC <- function(dps_test, densities, labels, classes){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    getAUC.R
  # Description: Script for evaluating the GRACE model (AUC)
  #              on longitudinal neuropsychological data.
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  require(pROC)
  #dps_test <- unlist(dps_test)
  dps_test <- as.vector(dps_test)
  #labels <- unlist(labels)
  labels <- as.vector(labels)
  mask <- is.na(labels)
  #dps_test <- dps_test[!mask]
  labels <- labels[!mask]
  dps_test <- dps_test[!is.na(dps_test)]
  
  mask_class <- labels %in% classes
  dps_test <- dps_test[mask_class]
  labels <- labels[mask_class]
  
  C <- length(classes)
  posteriors <- matrix(0, nrow = length(dps_test), ncol = C)
  
  priors <- list()
  for (c in seq(1, C))
    priors[c] <- sum(labels == classes[c]) / length(labels)
  
  for (i in 1:length(dps_test)) {
    point <- dps_test[i]
    likelihoods <- sapply(1:C, function(c) {
      predict(densities[[c]], x = point)
    })
    
    evidence <- sum(likelihoods * unlist(priors))
    
    for (c in seq(1, C)) {
      posteriors[i, c] <- (likelihoods[c] * priors[[c]]) / evidence
    }
  }
  
  labels <- match(labels, classes)
  if (C <= 2){
    roc_res <- roc(labels, posteriors[, 1])
    auc_res <- auc(roc_res)
  }
  else{
    aucs <- sapply(1:C, function(i){
      roc_res <- roc(labels == i, posteriors[, i])
      auc(roc_res)
    })
    
    auc_res <- mean(aucs)
  }
  
  return (list(auc_res = auc_res, priors_bayes = priors))
}