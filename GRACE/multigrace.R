setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("aux_grace/build_grace.R")
source("aux_grace/getDataADNI.R")
source("aux_grace/addNaNUniform.R")

multigrace <- function(){
# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    multigrace.R
# Description: Script for training the GRACE model
#              on longitudinal neuropsychological data.
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================
  require(openxlsx)
    require(foreach)
    require(slurmR)
    require(doParallel)
    require(parallel)
    require(qs)
    
    #adni_file <- "./data/ADNIMERGE_220706.csv"
    header <- "./data_multi/preclinical_CAM_jul22_"
    vector_range <- 2:5
    features_set <- c(1:12)
    btstrp <- 2
    perc_rem <- 0
    
    classes = c("CN", "MCI")
    classes_auc = c("CN", "MCI")
    
    markers_df <- data.frame(
      markers = c("RAVLT_forgetting", "RAVLT_immediate", "RAVLT_learning", "RAVLT_perc_forgetting", 
                  "ADAS13", "FAQ", "MMSE", "CDRSB", "LDELTOTAL", "mPACCdigit", "mPACCtrailsB", 
                  "TRABSCOR"),
      ascending = c(TRUE, FALSE, FALSE, TRUE, 
                    TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, 
                    TRUE),
      id = c(1, 2, 3, 4,
             5, 6, 7, 8, 9, 10, 11, 
             12)
    )
    
    feat_sub <- data.frame(matrix(nrow = 0, ncol = max(vector_range)))
    for (n in vector_range){
      aux_sub <- t(combn(markers_df[markers_df$id %in% features_set,]$markers, n))
      aux_sub <- cbind(aux_sub, matrix(NA, nrow = nrow(aux_sub), ncol = max(vector_range) - n))
      feat_sub <- rbind(feat_sub, aux_sub)
    }
    
    feat_sub <- cbind(feat_sub, matrix(NA, nrow = nrow(feat_sub), ncol = max(vector_range) * 2))
    feat_sub$dir_name <- NA
    
    feats <- markers_df$markers
    asc <- markers_df$ascending
    feats_selected <- markers_df[markers_df$id %in% features_set, "markers"]
    
    #Results table 
    results_table <- data.frame(matrix(nrow = nrow(feat_sub), ncol = max(vector_range)*2 + 24))
    feat_names <- paste0("feat_", 1:max(vector_range))
    feat_ids <- paste0("id_", 1:max(vector_range))
    colnames(results_table) <- c(feat_names, feat_ids, 
                                 "Corr_MCI_age_test", "Corr_MCI_reserve_test",
                                 "acc_scu_test", "acc_smci_test", "acc_pcu_test", 
                                 "AUC_test", "Corr_MCI_age_train", "Corr_MCI_reserve_train",
                                 "acc_scu_train", "acc_smci_train", "acc_pcu_train",
                                 "AUC_train",
                                 "std_corr_MCI_age_test", "std_corr_MCI_reserve_test", 
                                 "std_acc_scu_test", "std_acc_smci_test", "std_acc_pcu_test",
                                 "std_AUC_test", "std_corr_MCI_age_train", "std_corr_MCI_reserve_train",
                                 "std_acc_scu_train", "std_acc_smci_train", "std_acc_pcu_train",
                                 "std_AUC_train")
    
    #MAE table
    MAE_table_test <- data.frame(matrix(nrow = nrow(feat_sub), ncol = 2*length(features_set) + 1))
    MAE_table_train <- data.frame(matrix(nrow = nrow(feat_sub), ncol = 2*length(features_set) + 1))
    colnames(MAE_table_test) <- c("feat_id", markers_df[markers_df$id %in% features_set,]$markers,
                                  paste0("std_", markers_df[markers_df$id %in% features_set,]$markers))
    colnames(MAE_table_train) <- c("feat_id", markers_df[markers_df$id %in% features_set,]$markers,
                                   paste0("std_", markers_df[markers_df$id %in% features_set,]$markers))
    
    #Train-test split
    # df <- getDataADNI(adni_file)
    df <- read.csv("./data/df.csv")
    res_addNaNUniform <- addNaNUniform(df, feats_selected, perc_rem)
    df <- res_addNaNUniform$df
    res_prepareData <- prepareData(df, feats, asc)
    df_train <- res_prepareData$df_train
    df_test <- res_prepareData$df_test
    percent <- res_prepareData$percent
    point_feat <- res_prepareData$point_feat
    id_test <- res_prepareData$id_test

   for(i in 1:nrow(feat_sub)){
      #Create feat_sub table
      subs <- feat_sub[i, 1:max(vector_range)]
      match_idx <- match(subs, markers_df$markers)
      subs_asc <- markers_df$ascending[match_idx]
      subs_id <- markers_df$id[match_idx]
      feat_sub[i, (max(vector_range)+1):(max(vector_range) * 2)] <- subs_asc
      feat_sub[i, (max(vector_range) * 2 +1):(max(vector_range) * 3)] <- subs_id
      dir_name <- na.omit(subs_id)
      dir_name <- paste(dir_name, collapse = "_")
      dir_name <- paste0(header, dir_name)
      feat_sub[i, ]$dir_name <- dir_name
      #Create each directory with prepared data
      if(!dir.exists(dir_name)){
        dir.create(dir_name)
        cols <- c("ID", "AGE", "TIME", "ct","EXAMDATE", "M", "DX", "group", "DX_bl", "CONVERT")
        cols <- append(cols, subs[!is.na(subs)])
        df_subs <- df_train[, cols]
        all_markers_nan_idx <- apply(data.frame(df_subs[, subs[!is.na(subs)]]), 1, function(x) all(is.na(x)))
        df_subs <- df_subs[!all_markers_nan_idx, ]
        write.xlsx(df_subs, paste0(dir_name, "/dd.xlsx"))
      }
   }
  
     for (i in 1:nrow(feat_sub)){
      dir_name <- feat_sub[i, ]$dir_name
      file_in <- paste0(dir_name, "/dd.xlsx")
      save_model_path <- paste0(dir_name, "/grace_btstrp")
      
      if (file.exists(paste0(save_model_path, "_res.csv")))
        next
      
      else{
        feat <- c(t(na.omit(t(feat_sub[i, 1:max(vector_range)]))))
        asc <- c(t(na.omit(t(feat_sub[i, (max(vector_range) + 1):(max(vector_range) * 2)]))))
        id_feat <- c(t(na.omit(t(feat_sub[i, (max(vector_range) * 2 +1):(max(vector_range) * 3)]))))
        dd <- read.xlsx(paste0(dir_name, "/dd.xlsx"))
        percent_sub <- percent[id_feat]
        point_feat_sub <- point_feat[id_feat]
        id_train <- unique(dd$ID)
        #Clean test data
        all_markers_nan_idx <- apply(data.frame(df_test[, feat[!is.na(feat)]]), 1, function(x) all(is.na(x)))
        df_test.subs <- df_test[!all_markers_nan_idx,]
        id_test.subs <- unique(df_test.subs$ID)
        #Diagnose
        diagnose_train <- getDiagnose(dd)
        diagnose_test <- getDiagnose(df_test.subs)
        #Labels
        labels_train <- getSubVisitMatrix(dd, "DX")
        labels_test <- getSubVisitMatrix(df_test.subs, "DX")
        #Ages
        ages_train <- getSubVisitMatrix(dd, "TIME")
        ages_test <- getSubVisitMatrix(df_test.subs, "TIME")
        
        res_buildGrace <- buildGrace(dd, df_test.subs, percent_sub, point_feat_sub,
                                     id_train, id_test.subs, diagnose_train, 
                                     diagnose_test, labels_train, labels_test, 
                                     ages_train, ages_test, feat, btstrp, classes, 
                                     classes_auc, save_model_path)
        
        AUC_test <- res_buildGrace$AUC_test
        AUC_train <- res_buildGrace$AUC_train
        acc_scu_test <- res_buildGrace$acc_scu_test
        acc_smci_test <- res_buildGrace$acc_smci_test
        acc_pcu_test <- res_buildGrace$acc_pcu_test
        corr_MCI_age_test <- res_buildGrace$corr_MCI_age_test
        corr_MCI_reserve_test <- res_buildGrace$corr_MCI_reserve_test
        acc_scu_train <- res_buildGrace$acc_scu_train
        acc_smci_train <- res_buildGrace$acc_smci_train
        acc_pcu_train <- res_buildGrace$acc_pcu_train
        corr_MCI_age_train <- res_buildGrace$corr_MCI_age_train
        corr_MCI_reserve_train <- res_buildGrace$corr_MCI_reserve_train
        MAE_test <- res_buildGrace$MAE_test
        MAE_train <- res_buildGrace$MAE_train
        
        AUC_test_mean <- mean(AUC_test)
        AUC_test_std <- sd(AUC_test)
        AUC_train_mean <- mean(AUC_train)
        AUC_train_std <- sd(AUC_train)
        acc_scu_test_mean <- mean(acc_scu_test)
        acc_scu_test_std <- sd(acc_scu_test)
        acc_smci_test_mean <- mean(acc_smci_test)
        acc_smci_test_std <- sd(acc_smci_test)
        acc_pcu_test_mean <- mean(acc_pcu_test)
        acc_pcu_test_std <- sd(acc_pcu_test)
        corr_MCI_age_test_mean <- mean(corr_MCI_age_test)
        corr_MCI_age_test_std <- sd(corr_MCI_age_test)
        corr_MCI_reserve_test_mean <- mean(corr_MCI_reserve_test)
        corr_MCI_reserve_test_std <- sd(corr_MCI_reserve_test)
        acc_scu_train_mean <- mean(acc_scu_train)
        acc_scu_train_std <- sd(acc_scu_train)
        acc_smci_train_mean <- mean(acc_smci_train)
        acc_smci_train_std <- sd(acc_smci_train)
        acc_pcu_train_mean <- mean(acc_pcu_train)
        acc_pcu_train_std <- sd(acc_pcu_train)
        corr_MCI_age_train_mean <- mean(corr_MCI_age_train)
        corr_MCI_age_train_std <- sd(corr_MCI_age_train)
        corr_MCI_reserve_train_mean <- mean(corr_MCI_reserve_train)
        corr_MCI_reserve_train_std <- sd(corr_MCI_reserve_train)
        MAE_test_mean <- apply(MAE_test, 2, mean)
        MAE_test_std <- apply(MAE_test, 2, sd)
        MAE_train_mean <- apply(MAE_train, 2, mean)
        MAE_train_std <- apply(MAE_train, 2, sd)

        results_table[i,] <- c(feat_sub[i, 1:max(vector_range)],
                               feat_sub[i, (max(vector_range) * 2 +1):(max(vector_range) * 3)],
                               corr_MCI_age_test_mean,
                               corr_MCI_reserve_test_mean,
                               acc_scu_test_mean,
                               acc_smci_test_mean,
                               acc_pcu_test_mean,
                               AUC_test_mean,
                               corr_MCI_age_train_mean,
                               corr_MCI_reserve_train_mean,
                               acc_scu_train_mean,
                               acc_smci_train_mean,
                               acc_pcu_train_mean,
                               AUC_train_mean,
                               corr_MCI_age_test_std,
                               corr_MCI_reserve_test_std,
                               acc_scu_test_std,
                               acc_smci_test_std,
                               acc_pcu_test_std,
                               AUC_test_std,
                               corr_MCI_age_train_std,
                               corr_MCI_reserve_train_std,
                               acc_scu_train_std,
                               acc_smci_train_std,
                               acc_pcu_train_std,
                               AUC_train_std
                               )
        
        str_id <- paste(id_feat, collapse = "_")
        
        MAE_row_test_mean <- sapply(features_set, function(id) {
          if (!(id %in% id_feat)) {
            return("")
          } 
          else {
            return(MAE_test_mean[which(id_feat == id)])
          }
        })
        
        MAE_row_train_mean <- sapply(features_set, function(id) {
          if (!(id %in% id_feat)) {
            return("")
          } 
          else {
            return(MAE_train_mean[which(id_feat == id)])
          }
        })
        
        MAE_row_test_std <- sapply(features_set, function(id) {
          if (!(id %in% id_feat)) {
            return("")
          } 
          else {
            return(MAE_test_std[which(id_feat == id)])
          }
        })
        
        MAE_row_train_std <- sapply(features_set, function(id) {
          if (!(id %in% id_feat)) {
            return("")
          } 
          else {
            return(MAE_train_std[which(id_feat == id)])
          }
        })
        
        MAE_table_test[i,] <- c(str_id, 
                           MAE_row_test_mean,
                           MAE_row_test_std
                           )
        
        MAE_table_train[i,] <- c(str_id, 
                                MAE_row_train_mean,
                                MAE_row_train_std
                           )
      }
     }
    
    results_xlsx <- "preclinical_CAM_jul22_"
    results_xlsx <- paste0(results_xlsx, paste(markers_df$id, collapse = "_"))
    results_xlsx <- paste0(results_xlsx, ".xlsx")
    write.xlsx(results_table, results_xlsx)
    
    write.xlsx(MAE_table_test, "MAE_test_preclinical_CAM_jul22.xlsx")
    write.xlsx(MAE_table_train, "MAE_train_preclinical_CAM_jul22.xlsx")
    
    return (feat_sub)
}

multigrace()

