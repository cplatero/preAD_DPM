# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    conversion_time_days.R
# Description: Preprocess ADNI file
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================


#This function calculates the number of days since baseline for each visit
daysFromBaseline <- function(df){
  df$D <- 0
  #For each subject calculate days since baseline
  for (id in unique(df$ID)){
    subject <- df[df$ID == id, ]
    #Get baseline date
    baseline.date <- as.Date(head(subject$EXAMDATE, n=1), format="%Y/%m/%d")
    #Compute the difference between visit date and baseline date
    df[df$ID == id, ]$D = difftime(as.Date(subject$EXAMDATE, format="%Y/%m/%d"), 
                                   baseline.date,
                                   units="days")
  }
  df
}

#This function calculates conversion time in days for progressive subjects
conversionTimeDays <- function(df){
  
  #This subfunction calculates conversion time in days for a given subject
  computeTime <- function(subject){
    dx_bl <- head(subject$DX, n=1)
    #Iterate through each visit until baseline DX is different from visit DX
    for (ivisit in 1:nrow(subject)){
      visit <- subject[ivisit, ]
      if (dx_bl != visit$DX){
        #If the condition is met, exit the function and return the mean between current and previous
        #visit in days 
        return ((visit$D + subject[ivisit - 1, ]$D) / (2 * 365) )
      }
    }
  }
  
  df$CONVERSION_TIME_DAYS <- 0
  df$CONVERT <- 0
  #For each progressive subject, calculate conversion time in days
  for (id in unique(df$ID)){
    subject <- df[df$ID == id, ]
    
    if (nrow(subject) > 1 & (head(subject$DX, n=1) != tail(subject$DX, n=1))){
      df[df$ID == id, ]$CONVERSION_TIME_DAYS <- computeTime(subject)
      df[df$ID == id, ]$CONVERT <- 1
    }
    else df[df$ID == id, ]$CONVERT <- 0
  }
  
  df
}

centeredTimeDays <- function(df){
  df$onset <- 0 
  for (id in unique(df$ID)){
    subject <- df[df$ID == id, ]
    if(head(subject$DX, n=1) == "CN" & head(subject$CONVERT, n=1) == 1) #pCU
      df[df$ID == id, ]$onset <- (df[df$ID == id, ]$D - head(subject$CONVERSION_TIME_DAYS, n=1)*365)/30.4167
    else if (head(subject$DX, n=1) == "CN" & head(subject$CONVERT, n=1) == 0) #sCU
      df[df$ID == id, ]$onset <- (df[df$ID == id, ]$D - tail(subject$D, n=1))/30.4167
    else if ((head(subject$DX, n=1) == "MCI") & head(subject$CONVERT, n=1) == 0) #sMCI
      df[df$ID == id, ]$onset <- (df[df$ID == id, ]$D)/30.4167
  }
  return (df)
}




