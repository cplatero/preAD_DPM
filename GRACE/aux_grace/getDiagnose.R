getDiagnose <- function(df){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    getDiagnose.R
  # Description: First and last diagnoses
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  ids <- unique(df$ID)
  diagnose <- matrix(NA, nrow = 2, ncol = length(ids))
  
  for (idx in seq_along(ids)) {
    id <- ids[idx]
    df_sub <- df[df$ID == id, ]
    df_sub_dx <- df_sub$DX
    diagnose[1, idx] <- df_sub_dx[1]
    diagnose[2, idx] <- tail(df_sub_dx, 1)
  }
  
  return(diagnose)
}