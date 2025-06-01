# ============================================================
# Project:    Disease progression modeling from early AD stage
# Repository: https://github.com/cplatero/preAD_DPM
# Author:     Jorge Bengoa
# Email:      j.bpinedo@alumnos.upm.es
# Institution:Universidad Politécnica de Madrid
# ------------------------------------------------------------
# Filename:    grace_personalize.R
# Description: Script for personalizing with GRACE
#
# Version:    1.0
# Date:       2025-05-09
# Requires:   grace
# ============================================================

grace.personalize <- function(grace.model, df_test, markers){
  require(lme4)
  require(fda)

  dd <- data.frame(matrix(ncol = 4, nrow = 0))
  traindata <- do.call(rbind, lapply(grace.model$fits, function(x) x$subs))
  
  for(imarker in 1:length(markers)){
    marker <- markers[imarker]
    dd <- rbind(dd,
                cbind(id= df_test$ID,
                      t= df_test$ct,
                      Outcome= imarker,
                      y= df_test[, marker]))
  }
  
  dd <- dd[!(is.na(dd$y)),]
  
  #Calculate gamma0new
  dd.gamma0new <- lapply(unique(dd$Outcome), function(oc){
    dd.subs <- dd[dd$Outcome == oc, ]
    traindata.subs <- traindata[traindata$outcome == oc, ]
    
    gkinv <- with(traindata.subs, approxfun(ghat, argvals + gamma0, rule = 2))
    dd.subs$gamma0new <- with(dd.subs, gkinv(y) - t)
    dd.subs
  })
  
  dd <- do.call(rbind, lapply(dd.gamma0new, function(x) x))

  #Calculate gamma0
  shifts.train <- with(traindata, aggregate(gamma0new, by = list(id = id), 
                               FUN = mean, na.rm = TRUE))
  shifts <- with(dd, aggregate(gamma0new, by = list(id = id), FUN = mean, na.rm = TRUE))
  shifts$gamma0 <- shifts$x - mean(shifts.train$x)
  dd <- merge(dd[,], shifts[, c("id", "gamma0")], by="id", all.x = TRUE)
  
  #Calculate ghat
  dd.ghat <- lapply(unique(dd$Outcome), function(oc){
    dd.subs <- dd[dd$Outcome == oc, ]
    
    mfit <- grace.model$fits[[oc]]$monotone
    Wfd <- mfit$Wfdobj
    if(!is.null(Wfd)){
      beta <- mfit$beta
      dd.subs$ghat <- beta[1] + beta[2] * eval.monfd(with(dd.subs, t + gamma0), Wfd)
      dd.subs$smooth.resids <- (dd.subs$y - dd.subs$ghat)[,1]
    }
    else{
      coeff <- mfit$coefficients
      dd.subs$ghat <- coeff[1] + coeff[2] * (dd.subs$t + dd.subs$gamma0)
      dd.subs$smooth.resids <- dd.subs$y - dd.subs$ghat
    }
    
    dd.subs
  })
  
  dd <- do.call(rbind, lapply(dd.ghat, function(x) x))

  #Calculata alpha0 and alpha1
  dd.alphas <- lapply(unique(dd$id), function(id){
    dd.id <- dd[dd$id == id, ]

    dd.id.alphas <- lapply(unique(dd.id$Outcome), function(oc){
      dd.id.oc <- dd.id[dd.id$Outcome == oc, ]
      nvisits <- nrow(na.omit(dd.id.oc))
      
      if (nvisits == 0){ 
        dd.id.oc$alpha0 <- 0
        dd.id.oc$alpha1 <- 0
      }
      else if (nvisits == 1){
        dd.id.oc$alpha0 <- with(dd.id.oc, smooth.resids)
        dd.id.oc$alpha1 <- 0
      }
      else{
        dd.id.oc.filtered <- na.omit(dd.id.oc)
        x <- with(dd.id.oc.filtered, t)
        y <- with(dd.id.oc.filtered, smooth.resids)
        model <- lm(y ~ x)
        dd.id.oc$alpha0 <- coef(model)[1]
        dd.id.oc$alpha1 <- coef(model)[2]
      }
      dd.id.oc
    })
    dd.id <- do.call(rbind, lapply(dd.id.alphas, function(x) x))
    dd.id
  })
  
  dd <- do.call(rbind, lapply(dd.alphas, function(x) x))
  
  return (dd)
}

# setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# 
# library(openxlsx)
# markers <- c("RAVLT_learning", "ADAS13", "CDRSB", "TRABSCOR")
# df = read.csv("./data/prepared_data.csv")
# df <- trainTestData(df)
# df_train <- df$train
# df_test <- df$test
# model <- readRDS("./data/model1.rds")

# grace.personalize(model,df_test, markers)