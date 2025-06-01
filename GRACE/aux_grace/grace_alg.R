# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    grace_alg
# Description: Script for training with GRACE
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

# This function takes a dataframe with processed data and fits it using grace algorithm.
#   - Inputs:
#     - df: train dataframe with processed data 
#     - markers: markers to study
#     - save: TRUE for saving the resulting dataframe
#   - Outputs:
#     - dd1: dataframe with grace fits
#     - grace_model: object returned by grace algorithm with fitting information
graceFit <- function(df, markers, save= TRUE, save_path = "./data/graceFit.RData" ){
  require(grace)
  #require(rhdf5)
  #Create dd dataframe
  dd <- data.frame(matrix(ncol = 4, nrow = 0))
  colnames(dd) <- c("id", "t", "y", "Outcome")
  #Fill dd with data for each marker
  for(imarker in 1:length(markers)){
    marker <- markers[imarker]
    dd <- rbind(dd,
                cbind(id= df$ID,
                      t= df$ct,
                      Outcome= imarker,
                      y= df[, marker]))
  }
  #Drop NA values in y
  dd <- dd[!(is.na(dd$y)),]
  #Apply grace algorithm 
  grace.simulation.fits <- with(dd, grace(t,y, Outcome, id, plots = FALSE))
  #Get fits and combine all data to a single dataframe
  dd1 <- do.call(rbind, lapply(grace.simulation.fits$fits, function(x) x$subs))
  dd1$Outcome <-  factor(dd1$outcome, levels = 1:4)
  dd1 <- dd1[with(dd1, order(id)), ]
  dd1<- dd1[with(dd1, order(id, outcome, argvals)), ]
  
  
  #Save dd1 and model
  if(save){
    saveRDS(grace.simulation.fits, save_path)
  }
  
  return (list(dd1 = dd1, grace_model = grace.simulation.fits))
}
