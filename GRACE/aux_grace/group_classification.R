# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    group_classification.R
# Description: Preprocess ADNI file
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

#Return subjects classified by clinical group (DX_bl)
preclinicalGroups <- function(df, save=TRUE){
  #require(rhdf5)
  #CN
  cn <- df[df$DX_bl == "CN", ]
  cn <- subset(cn, !is.na(DX))
  cn$DX_bl0 <- cn$DX_bl
  #SMC
  smc <- df[df$DX_bl == "SMC", ]
  smc <- subset(smc, !is.na(DX))
  smc$DX_bl0 <- replace(smc$DX_bl, smc$DX_bl == "SMC", "CN")
  #MCI
  mci <- df[df$DX_bl == "EMCI" | df$DX_bl == "LMCI", ]
  mci <- subset(mci, !is.na(DX))
  mci$DX_bl0 <- replace(mci$DX_bl, mci$DX_bl == "EMCI" | mci$DX_bl == "LMCI", "MCI")
  #Save data
  # if(save){
  #   path <- "./data/preclinicalGroups.h5"
  #   if(file.exists(path))
  #     file.remove(path)
  #   h5createFile(path)
  #   h5write(cn, file= path, "cn")
  #   h5write(smc, file= path, "smc")
  #   h5write(mci, file= path, "mci")
  #   h5closeAll()
  # }
  return (list(cn= cn, smc= smc, mci= mci))
}

#Calculate conversion time for a subject
convertTime <- function(df)
{
  for (ivisit in 1:nrow(df))
  {
    visit <- df[ivisit, ]
    #If visit DX is different from visit DX_bl 
    if (visit$DX_bl0 != visit$DX)
    {
      #If VISCODE is bl, conversion time is 0 (not going to happen)
      if(visit$VISCODE == "bl")
        return (0)
      #Else calculate time since baseline in years and exit the function
      else 
        return (as.numeric(visit$M)/12.0)
    }
  }
}

#Return subjects classified by estable and progressive within a clinical group
estableProgressive <- function(df)
{
  df$CENSURE_TIME <- 0
  df$CONVERT_TIME <- 0
  #Classify by progressive and stable
  progressive <- data.frame(matrix(ncol = ncol(df), nrow=0))
  estable <- data.frame(matrix(ncol = ncol(df), nrow=0))
  colnames(progressive) <- colnames(df)
  colnames(estable) <- colnames(df)
  #For each RID compare group at baseline and group after last observable visit 
  rids <- unique(df$RID)
  for (rid in rids)
  {
    #Filter dataframe by id
    filtered_df <- df[df$RID == rid, ]
    #If filtered_df is empty, no visits were registered from baseline visit on
    if (nrow(filtered_df) > 1)
    {
      #Censure time - time to last visit
      if (tail(filtered_df$VISCODE, n=1) != "bl")
      {
        filtered_df$CENSURE_TIME <- as.numeric(tail(filtered_df$M, n=1)) /12.0
      }
      #Look for last observable visit (drop nan values) and compare it to baseline (filtered_df is chronologically ordereded)
      #If both values are equal the subject is stable
      if ((head(filtered_df$DX_bl0, n=1) == tail(filtered_df$DX, n=1)) |
          (head(filtered_df$DX_bl0, n=1) == "MCI" & tail(filtered_df$DX, n=1) == "CN"))
      {
        estable <- rbind(estable, filtered_df)
      }
      else 
      {
        filtered_df$CONVERT_TIME <- convertTime(filtered_df)
        progressive <- rbind(progressive, filtered_df)
      }
    }
    #If no more visits were registered the subject is stable
    else 
    {
      #Censure time remains 0
      estable <- rbind(estable, filtered_df)
    }
  }
  return (list(estable=estable, progressive=progressive))
}