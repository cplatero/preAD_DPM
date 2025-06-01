getTrueOnset <- function(labels_test, ages_test) {
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    getTrueOnset.R
  # Description: Script for calculating conversion time
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  I <- nrow(labels_test)
  true_onset <- matrix(NA, nrow = 1, ncol = I)
  MCI <- ifelse(labels_test == "MCI", 1, 0)
  
  for (i in 1:I) {
    if (sum(!is.na(MCI[i, ])) > 0) {
      mask_values <- !is.na(ages_test[i, ])  
      age_subj <- ages_test[i, mask_values]
      label_subj <- MCI[i, mask_values]
      idx <- which(label_subj == 1)
      
      if (label_subj[length(label_subj)]) {
        if (idx[1] > 1) {
          true_onset[1, i] <- (age_subj[idx[1]] + age_subj[idx[1] - 1]) / 2
        } else {
          true_onset[1, i] <- age_subj[idx[1]]
        }
      }
    }
  }
  
  return(true_onset)
}