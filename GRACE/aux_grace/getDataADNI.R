source("aux_grace/conversion_time_days.R")
source("aux_grace/percentile_interp.R")
source("aux_grace/train_test_data.R")

# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    getDataADNI.R
# Description: Preprocess ADNI file
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

centeredTime <- function(df){
  df$ct <- 0
  #Calculate centered time in yeas for every id
  for (id in unique(df$ID))
  {
    subject <- df[df$ID == id, ]
    ti1 <- head(subject$M, n=1)
    tiend <- tail(subject$M, n=1)
    subject$ct <- subject$M -(ti1 + tiend)/2
    subject$ct <- subject$ct/12.0
    df[df$ID == id, ]$ct <- subject$ct
  }
  
  df
}

dropOneVisitSubjects <- function(df){
  one_visit_ids <- c()
  #Get ids for subjects with one visit
  for (id in unique(df$ID)){
    subject <- df[df$ID == id, ]
    if(nrow(subject) == 1)
      one_visit_ids <- append(one_visit_ids, subject$ID)
  }
  #Filter dataframe
  df <- df[!(df$ID %in% one_visit_ids), ]
  
  df
}

getDataADNI <- function(path){
  df <- read.csv(path)
  df <- df[!is.na(df$AGE) & !df$DX == "", ]
  names(df)[names(df) == "RID"] <- "ID"
  
  ids_ad <- df$ID[(df$DX_bl == "AD") | (df$VISCODE == "bl" & df$DX == "Dementia")]
  df <- df[!df$ID %in% ids_ad, ]
  
  df$TIME <- df$AGE + df$M/12
  
  df <- centeredTime(df)
  df <- daysFromBaseline(df)
  df <- conversionTimeDays(df)
  df <- centeredTimeDays(df)
  
  df$group <- 1
  ids_cn <- df[(df$DX == "CN") & (df$M == 0), ]$ID
  df[df$ID %in% ids_cn, ]$group <- 0
  
  # scu, pcu, smci
  mask <- (df$group == 0) | ((df$group == 1) & (df$CONVERT == 0))
  df <- df[mask,]
  
  mask_ad <- df$DX == "Dementia"
  df <- df[!mask_ad, ]
  
  # Drop one visit subjects
  df <- dropOneVisitSubjects(df)
  
  df
}

prepareData <- function(df, feats, asc){
  cols <- c("ID", "TIME", "AGE", "DX", "EXAMDATE", "M", "D", "group", 
            "CONVERT", "CONVERSION_TIME_DAYS", "onset", "ct", "DX_bl", feats)
  
  for (i in 1:length(feats))
    if (asc[i] == FALSE)
      df[feats[i]] <- -df[feats[i]]
  
  feats_orig <- paste0(feats, "_orig")
  df <- df[, cols]
  df[, feats_orig] <- df[, feats]
  
  ids <- unique(df$ID)
  data_split <- trainTestData(df)
  df_train <- data_split$train
  df_test <- data_split$test
  id_test <- unique(df_test$ID)
  
  group_train <- ifelse(df_train$group == 0 & df_train$CONVERSION_TIME_DAYS == 0, 0,
                        ifelse(df_train$group == 1 & df_train$CONVERSION_TIME_DAYS == 0, 1,
                               ifelse(df_train$group == 0 & df_train$CONVERSION_TIME_DAYS > 0, 2, NA)))
  group_train <- data.frame(group_train)
  
  group_test <- ifelse(df_test$group == 0 & df_test$CONVERSION_TIME_DAYS == 0, 0,
                       ifelse(df_test$group == 1 & df_test$CONVERSION_TIME_DAYS == 0, 1,
                              ifelse(df_test$group == 0 & df_test$CONVERSION_TIME_DAYS > 0, 2, NA)))
  group_test <- data.frame(group_test)
  
  features_train <- df_train[, feats]
  colnames(features_train) <- seq_along(feats)
  
  features_test <- df_test[, feats]
  colnames(features_test) <- seq_along(feats)
  
  res_percentile_interp <- percentile_interp(features_train, group_train)
  percent <- res_percentile_interp$percent
  point_feat <- res_percentile_interp$point_feat
  feat2per_train <- res_percentile_interp$feat2per
  
  res_percentile_interp <- percentile_interp(features_test, group_test, percent, point_feat)
  feat2per_test <- res_percentile_interp$feat2per
  
  df_train[, feats] <- feat2per_train
  df_test[, feats] <- feat2per_test
  
  return(list(df_train = df_train, df_test = df_test, percent = percent, 
              point_feat = point_feat, id_test = id_test))
}
