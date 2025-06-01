trainTestData <- function(df, test_proportion= 0.2){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    train_test_data.R
  # Description: Script for splitting data into train and test
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  set.seed(42)
  
  ids <- unique(df$ID)
  test_idx <- sample(length(ids), floor(length(ids) * test_proportion))
  test_ids <- ids[test_idx]
  
  test_df <- df[df$ID %in% test_ids, ]
  train_df <- df[!(df$ID %in% test_ids), ]
  
  return (list(train=train_df, test=test_df))
}
