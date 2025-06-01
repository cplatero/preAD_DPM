source("aux_grace/getTrueOnset.R")

getOnset <- function(DPS_test, priors_bayes, KDE, ages_test, labels_test, btstrp){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    getOnset.R
  # Description: Script for evaluating the GRACE model (clinical scores)
  #              on longitudinal neuropsychological data.
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  I <- dim(DPS_test)[2]
  C <- 2
  prob_MCI <- array(NA, dim = dim(DPS_test))
  ages_onset <- matrix(NA, nrow = btstrp, ncol = I)

  for (n in 1:btstrp) {
    prior <- priors_bayes[[n]]
    kde <- KDE[[n]]
    dps_test <- DPS_test[n, , ]
    for (i in 1:nrow(dps_test)) {
      dps_subj <- dps_test[i, ]
      mask_values <- !is.na(dps_subj)
      dps_subj <- dps_subj[mask_values]
      
      likelihoods <- sapply(1:C, function(c) {
        predict(kde[[c]], x = dps_subj)
      })

      likelihoods <- if(is.matrix(likelihoods)) likelihoods else t(as.matrix(likelihoods))
      evidence <- rowSums(likelihoods * unlist(prior))
      prob_MCI[n, i, mask_values] <- (likelihoods[,2] * prior[[2]]) / evidence #Posterior MCI
    }
  }
 
  
  for (n in 1:btstrp) {
    dps_test_n <- DPS_test[n, , ]
    prob_MCI[n, , ] <- ifelse(!is.na(prob_MCI[n, , ]), 
                              ifelse(prob_MCI[n, , ] > 0.5, 1.0, 0.0), 
                              NA)
    
    for (i in 1:I) {
      if (sum(!is.na(prob_MCI[n, i, ])) > 0) {
        mask_values <- !is.na(ages_test[i, ]) 
        age_subj <- ages_test[i, mask_values]
        label_subj <- prob_MCI[n, i, mask_values]
        idx <- which(label_subj == 1)
        if (label_subj[length(label_subj)]) {
          if (idx[1] > 1) {
            ages_onset[n, i] <- (age_subj[idx[1]] + age_subj[idx[1] - 1]) / 2
          } else {
            ages_onset[n, i] <- age_subj[idx[1]]
          }
        }
      }
    }
  }
  
  true_onset <- getTrueOnset(labels_test, ages_test)
  
  # Metrics
  ages_bsl <- ages_test[, 1]
  
  acc_scu <- matrix(NA, nrow = btstrp, ncol = 1)
  acc_smci <- matrix(NA, nrow = btstrp, ncol = 1)
  acc_pcu <- matrix(NA, nrow = btstrp, ncol = 1)
  corr_MCI_age <- matrix(NA, nrow = btstrp, ncol = 1)
  corr_MCI_reserve <- matrix(NA, nrow = btstrp, ncol = 1)
  
  n_scu <- sum(is.na(true_onset))
  n_smci <- sum(na.omit(t(true_onset == ages_bsl)))

  for (n in 1:btstrp) {
    acc_scu[n, ] <- 100 * sum(is.na(true_onset) & is.na(ages_onset[n, ])) / n_scu
    acc_smci[n, ] <- 100 * sum(na.omit(t((ages_onset[n, ] == ages_bsl) & (true_onset == ages_bsl)))) / n_smci
  }
  
  n_pcu <- sum(na.omit(t(true_onset > ages_bsl)))
  
  for (n in 1:btstrp) {
    acc_pcu[n, ] <- 100 * sum(na.omit(t((ages_onset[n, ] > ages_bsl) & (true_onset > ages_bsl)))) / n_pcu
    
    corr_data_age <- data.frame(cbind(ages_onset[n, ], true_onset[1, ]))
    corr_data_age <- corr_data_age[complete.cases(corr_data_age), ]
    corr_MCI_age[n, ] <- cor(corr_data_age)[1, 2]
    
    corr_data_reserve <- data.frame(cbind(ages_onset[n, ] - ages_bsl, true_onset[1, ] - ages_bsl))
    corr_data_reserve <- corr_data_reserve[complete.cases(corr_data_reserve), ]
    corr_MCI_reserve[n, ] <- cor(corr_data_reserve)[1, 2]
  }
  
  return(list(acc_scu = acc_scu, acc_smci = acc_smci, acc_pcu = acc_pcu, 
              corr_MCI_age = corr_MCI_age, corr_MCI_reserve = corr_MCI_reserve))
}