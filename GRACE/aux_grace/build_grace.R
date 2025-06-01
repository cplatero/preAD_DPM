source("aux_grace/build_model_grace.R")
source("aux_grace/kde.R")
source("aux_grace/getAUC.R")
source("aux_grace/getOnset.R")
source("aux_grace/getMae.R")

buildGrace <- function(df_train, df_test, percent, point_feat, id_train, id_test, 
                       diagnose_train, diagnose_test, labels_train, labels_test, 
                       ages_train, ages_test, markers, btstrp, classes, classes_auc, 
                       save_model_path){
  # ============================================================
  # Project:    Disease progression modeling from early AD stage
  # Repository: https://github.com/cplatero/preAD_DPM
  # Author:     Jorge Bengoa
  # Email:      j.bpinedo@alumnos.upm.es
  # Institution:Universidad Politécnica de Madrid
  # ------------------------------------------------------------
  # Filename:    build_grace.R
  # Description: Building DPMs using GRACE
  #
  # Version:    1.0
  # Date:       2025-05-09
  # Requires:   grace
  # ============================================================
  
  res_buildModelGrace <- buildModelGrace(df_train, markers, btstrp, diagnose_train, 
                                         classes, save_model_path)
  
  Id_train <- res_buildModelGrace$Id_train
  Idx_train <- res_buildModelGrace$Idx_train
  Gamma0 <- res_buildModelGrace$Gamma0
  DPS_train <- res_buildModelGrace$DPS_train
  
  n_test <- length(id_test)
  n_train <- length(id_train)
  num_points_test <- table(df_test$ID)[as.character(id_test)]
  num_points_train <- table(df_train$ID)[as.character(id_train)]
  max_visit_test <- max(num_points_test)
  max_visit_train <- max(num_points_train)
  Gamma_test <- matrix(0, nrow = btstrp, ncol = n_test)
  DPS_test <- array(NA, dim = c(btstrp, n_test, max_visit_test))
  MAE_test <- array(NA, dim = c(btstrp, length(markers)))
  Gamma_train_all <- matrix(0, nrow = btstrp, ncol = n_train)
  DPS_train_all <- array(NA, dim = c(btstrp, n_train, max_visit_train))
  MAE_train <- array(NA, dim = c(btstrp, length(markers)))
  
  for (n in seq(1, btstrp)){
    path <- paste0(save_model_path, n, ".rds")
    model <- readRDS(path)
    
    dd <- grace.personalize(model, df_test, markers)
    Gamma_test[n, ] <- dd[!duplicated(dd$id), "gamma0"]
    MAE_test[n, ] <- getMaeGrace(dd, percent, point_feat, markers)
    
    dd_train <- grace.personalize(model, df_train, markers)
    Gamma_train_all[n, ] <- dd_train[!duplicated(dd_train$id), "gamma0"]
    MAE_train[n, ] <- getMaeGrace(dd_train, percent, point_feat, markers)
    
    for (i in 1:length(id_test)) {
      id <- id_test[i]
      tijc <- df_test[df_test$ID == id, "ct"]
      gamma0 <- Gamma_test[n,i]
      dps <- tijc + gamma0
      DPS_test[n, i, 1:num_points_test[i]] <- dps
    }
    
    for (i in 1:length(id_train)) {
      id <- id_train[i]
      tijc <- df_train[df_train$ID == id, "ct"]
      gamma0 <- Gamma_train_all[n,i]
      dps <- tijc + gamma0
      DPS_train_all[n, i, 1:num_points_train[i]] <- dps
    }
  }
  
  KDE <- list()
  
  for (n in seq(1, btstrp))
    KDE[[n]] <- densityKDE(DPS_train[n,,], labels_train[Idx_train[n,], ], classes)
  #AUC test
  AUC_test <- rep(0, btstrp)
  priors_bayes <- list()
  for (n in seq(1, btstrp)){
    res_getAUC<- getAUC(DPS_test[n,,], KDE[[n]], labels_test, classes_auc)
    AUC_test[n] <- res_getAUC$auc_res
    priors_bayes[[n]] <- res_getAUC$priors_bayes
  }
  
  #AUC train
  AUC_train <- rep(0, btstrp)
  for (n in seq(1, btstrp))
    AUC_train[n] <- getAUC(DPS_train_all[n,,], KDE[[n]], 
                           labels_train, classes_auc)$auc_res

  res_getOnset <- getOnset(DPS_test, priors_bayes, KDE, ages_test, labels_test, btstrp)

  acc_scu_test <- res_getOnset$acc_scu
  acc_smci_test <- res_getOnset$acc_smci
  acc_pcu_test <- res_getOnset$acc_pcu
  corr_MCI_age_test <- res_getOnset$corr_MCI_age
  corr_MCI_reserve_test <- res_getOnset$corr_MCI_reserve
  
  res_getOnset <- getOnset(DPS_train_all, priors_bayes, KDE, ages_train, labels_train, btstrp)
  
  acc_scu_train <- res_getOnset$acc_scu
  acc_smci_train <- res_getOnset$acc_smci
  acc_pcu_train <- res_getOnset$acc_pcu
  corr_MCI_age_train <- res_getOnset$corr_MCI_age
  corr_MCI_reserve_train <- res_getOnset$corr_MCI_reserve
  
  data <- list(Id_train = Id_train,
               Idx_train = Idx_train,
               Gamma_train = Gamma0,
               DPS_train = DPS_train,
               DPS_train_all = DPS_train_all,
               labels_train = labels_train,
               ages_train = ages_train,
               id_test = id_test,
               Gamma_test = Gamma_test,
               DPS_test = DPS_test,
               labels_test = labels_test,
               ages_test = ages_test,
               btstrp = btstrp,
               percent = percent, 
               point_feat = point_feat,
               df_train = df_train,
               df_test = df_test)
  
  qsave(data, file=paste0(save_model_path, ".qs"))
  
  
  return(list(AUC_test = AUC_test,
              AUC_train = AUC_train,
              acc_scu_test = acc_scu_test,
              acc_smci_test = acc_smci_test,
              acc_pcu_test = acc_pcu_test,
              corr_MCI_age_test = corr_MCI_age_test,
              corr_MCI_reserve_test = corr_MCI_reserve_test,
              acc_scu_train = acc_scu_train,
              acc_smci_train = acc_smci_train,
              acc_pcu_train = acc_pcu_train,
              corr_MCI_age_train = corr_MCI_age_train,
              corr_MCI_reserve_train = corr_MCI_reserve_train,
              MAE_test = MAE_test,
              MAE_train = MAE_train))
}




