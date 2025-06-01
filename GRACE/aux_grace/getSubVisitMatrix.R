getSubVisitMatrix <- function(df, col) {
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    getSubVisitMatrix.R
  # Description: Dataframe to matrix for a given column
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  ids <- unique(df$ID)
  dx_per_sub <- split(df[[col]], df$ID)
  ncol <- max(sapply(dx_per_sub, length))
  
  col_matrix <- matrix(NA, nrow = length(ids), ncol = ncol)
  
  for (idx in seq_along(ids)) {
    id <- ids[idx]
    dx_sub <- dx_per_sub[[as.character(id)]]
    col_matrix[idx, 1:length(dx_sub)] <- dx_sub
  }
  
  return(col_matrix)
}