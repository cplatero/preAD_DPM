source("aux_grace/grace_alg.R")
source("aux_grace/train_test_data.R")
source("aux_grace/grace_personalize.R")
source("aux_grace/getDiagnose.R")
source("aux_grace/getSubVisitMatrix.R")

buildModelGrace <- function(df_train, markers, btstrp, diagnose, classes, save_model_path){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    build_model_grace.R
  # Description: Building DPMs using GRACE
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  ids <- unique(df_train$ID)
  n_train <- length(ids)
  num_points <- sapply(ids, function(id) sum(df_train$ID == id))
  max_visits <- max(num_points)
  Idx_train <- matrix(0, nrow = btstrp, ncol = n_train)
  Id_train <- matrix(0, nrow = btstrp, ncol = n_train)
  Gamma0 <- matrix(0, nrow=btstrp, ncol = n_train)
  DPS_train <- array(NA, dim = c(btstrp, n_train, max_visits))
  
  num_points <- sapply(ids, function(id) sum(df_train$ID == id))
  
  for (n in 1:btstrp) {
    idx_btstrp <- c()
    set.seed(n)
    
    for (u in seq_along(classes)) {
      for (uu in seq_along(classes)) {
        idx_u <- which(diagnose[1,] == classes[u] & diagnose[2,] == classes[uu])
        med_u <- median(num_points[idx_u])
        idx_many <- idx_u[num_points[idx_u] > med_u]
        
        if (length(idx_many) > 0) {
          idx_rnd <- sample(idx_many, size = length(idx_many), replace = TRUE)
          idx_btstrp <- c(idx_btstrp, idx_rnd)
        }
        
        idx_few <- idx_u[num_points[idx_u] <= med_u]
        
        if (length(idx_few) > 0) {
          idx_rnd <- sample(idx_few, size = length(idx_few), replace = TRUE)
          idx_btstrp <- c(idx_btstrp, idx_rnd)
        }
      }
    }
    
    Idx_train[n, ] <- idx_btstrp
    Id_train[n, ] <- ids[idx_btstrp]
    
    #Train
    path <- paste0(save_model_path, n, ".rds")
    df_train_btstrp <- df_train[df_train$ID %in% Id_train[n, ], ]
    
    if (!file.exists(path)) {
      fit_res <- graceFit(df_train_btstrp, markers, save = TRUE, save_path = path)
      model <- fit_res$grace_model
      dd1 <- fit_res$dd1
    } 
    else {
      model <- readRDS(path)
      
      dd1 <- do.call(rbind, lapply(model$fits, function(x) x$subs))
      dd1$Outcome <-  factor(dd1$outcome, levels = 1:4)
      dd1 <- dd1[with(dd1, order(id)), ]
      dd1<- dd1[with(dd1, order(id, outcome, argvals)), ]
    }
    
    gamma0 <- dd1[!duplicated(dd1$id), "gamma0"] 
    id_gamma <- dd1[!duplicated(dd1$id), "id"]
    Gamma0[n, ] <- gamma0[match(Id_train[n,], id_gamma)]
    
    for (i in 1:length(Idx_train[n, ])) {
      idx <- Idx_train[n, i]
      id <- Id_train[n, i]
      tijc <- df_train[df_train$ID == id, "ct"]
      gamma0 <- Gamma0[n, i]
      dps <- tijc + gamma0
      DPS_train[n, i, 1:num_points[idx]] <- dps
    }
  }
  
  return(list(Id_train = Id_train, Idx_train = Idx_train, 
              Gamma0 = Gamma0, DPS_train = DPS_train))
}
