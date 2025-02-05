.libPaths("path")
library(glmnet)
library(lassoshooting)
library(mvtnorm)
library(pROC)
library(rmutil)
library(glmnet)
library(lassoshooting)
library(mvtnorm)
library(pROC)
library(rmutil)
library(grplasso)
library(Matrix)

M <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
sim=M

load('eMERGE_extreme_obesity.RData')
ob_data$SEX = ob_data$SEX - 1

logistic<-function(x){
  1/(1+exp(-x))
}

ST.init<-function(X.tar,y.tar){ 
  #single-task initialization
  p <- ncol(X.tar)
  n0.tar <- length(y.tar)
  fit <- cv.glmnet(x=X.tar, y=y.tar, nfolds = 5, family='binomial')
  lam.const = fit$lambda.min / sqrt(2*log(p)/n0.tar)
  beta0 = c(fit$glmnet.fit$a0[which(fit$lambda == fit$lambda.min)], fit$glmnet.fit$beta[, which(fit$lambda == fit$lambda.min)])
  return(list(beta0=beta0, lam.const=lam.const))
}

TL.init<-function(X.tar, y.tar, X.src, y.src, w=NULL){
  p <- ncol(X.tar)
  n0.src <- length(y.src)
  n0.tar <- length(y.tar)
  
  if(is.null(w)){
    #source population estimates
    fit.src <- cv.glmnet(x=X.src, y=y.src, family='binomial', nfolds=5)
    lam.const = fit.src$lambda.min / sqrt(2*log(p)/n0.src)
    w0 <- c(fit.src$glmnet.fit$a0[which(fit.src$lambda == fit.src$lambda.min)], fit.src$glmnet.fit$beta[, which(fit.src$lambda == fit.src$lambda.min)])
  }else{
    w0 = w
  }
  #target population estimates
  fit.tar <- cv.glmnet(x=X.tar, y=y.tar, nfolds=5, family='binomial', offset=w0[1]+X.tar%*%w0[-1])
  delta0 = c(fit.tar$glmnet.fit$a0[which(fit.tar$lambda == fit.tar$lambda.min)], fit.tar$glmnet.fit$beta[, which(fit.tar$lambda == fit.tar$lambda.min)])
  return(list(w0=w0, delta0=delta0, beta0=w0+delta0))
}

thres<-function(b, k, p){
  b*(rank(abs(b))>=(p-k))
}


Trans.global<-function(X.tar, y.tar, X.src, y.src, delta=NULL){
  p <- ncol(X.tar)
  n0.tar <- length(y.tar)
  K <- length(y.src)  ###(K>1)
  ##global method
  Xdelta = c()
  if(is.null(delta)){
    for(k in 1:K){
      w.k = as.numeric(ST.init(X.src[[k]], y.src[[k]])$beta0)
      fit.tar <- cv.glmnet(x=X.tar, y=y.tar, nfolds=5, family='binomial', offset=w.k[1]+X.tar%*%w.k[-1], lambda=seq(0.25, 0.05, length.out=20)*sqrt(2*log(p)/n0.tar))
      delta.k = c(fit.tar$glmnet.fit$a0[which(fit.tar$lambda == fit.tar$lambda.min)], fit.tar$glmnet.fit$beta[, which(fit.tar$lambda == fit.tar$lambda.min)])
      delta.k.thre = thres(delta.k, sqrt(n0.tar), p) ###+threshold
      Xdelta.k = tcrossprod(delta.k.thre[-1], X.src[[k]])+delta.k.thre[1]
      Xdelta = c(Xdelta, Xdelta.k)
    }
  }else{
    for(k in 1:K){
      Xdelta.k = tcrossprod(delta[[k]][-1], X.src[[k]])+delta[[k]][1]
      Xdelta = c(Xdelta, Xdelta.k)
    }
  }
  
  XX.src <- yy.src <- NULL
  for(k in 1:K){
    XX.src <- rbind(XX.src, X.src[[k]])
    yy.src <- c(yy.src, y.src[[k]])
  }
  
  n0.tar.global <- length(c(y.tar,yy.src))
  offset <- c(rep(0, nrow(X.tar)), -Xdelta)
  fit.global <- cv.glmnet(x=rbind(X.tar,XX.src), y=c(y.tar,yy.src), nfolds=5, family='binomial', offset=offset, lambda=seq(0.25, 0.05, length.out=20)*sqrt(2*log(p)/n0.tar.global))
  beta.hat = c(fit.global$glmnet.fit$a0[which(fit.global$lambda == fit.global$lambda.min)], fit.global$glmnet.fit$beta[, which(fit.global$lambda == fit.global$lambda.min)])
  
  return(beta.hat)
}

Agg.fun<-function(B, X.til, y.til, const=2){
  X.til = cbind(1, X.til)
  loss.B <- apply(B, 2, function(b) - sum(y.til*log(logistic(X.til%*%b))+(1-y.til)*log(1-logistic(X.til%*%b))))
  eta.hat <- exp(-const*loss.B)/sum(exp(-const*loss.B))
  return(eta.hat)
}

Agg.fun.new<-function(B, X.til, y.til, const=2){
  X.til = cbind(1, X.til)
  XX = X.til%*%B
  eta.hat = glm(y.til~XX-1, family = binomial(link = "logit"))$coefficients
  #eta.hat = glm(y.til~XX-1, family = gaussian)$coefficients
  return(eta.hat)
}


rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}


mse.fun<- function(beta.true, est, X.test=NULL){
  mean((beta.true[-1]-est[-1])^2)
}

get.auc = function(X, y, beta){
  pred.y = logistic(X%*%beta)
  #mean(abs(y.test1-pred.y))
  pROC::auc(pROC::roc(as.numeric(y), as.numeric(pred.y), warning=F), warning=F)
}


create.synthetic <- function(K, X.tar, n.src, r, B){
  n.syn = round(n.src*r)     #total number of synthetic data for each source: r*source sample
  
  X.syn <- y.syn <- list()
  for(k in 1:K){
    print(k)
    
    X.syn.k = X.tar[sample(1:nrow(X.tar), size = n.syn[k], replace=TRUE),]
    y.syn.k = rbinom(n.syn[k], 1, prob = logistic(c(cbind(1,X.syn.k) %*% B[[k]])))
    
    X.syn[[k]] = X.syn.k
    y.syn[[k]] = y.syn.k
  }
  return(list(X.syn=X.syn, y.syn=y.syn))
}



simu <- function(X.tar, y.tar, X.src, y.src, p, K, n0, nt, r, X.til, y.til, X.test1, y.test1, X.test2, y.test2){
  n.tar <- n0 
  n.src <- nt 
  
  ###target only
  fit0 <- ST.init(X.tar, y.tar)
  beta.tar = as.numeric(fit0$beta0)
  get.auc(X.test1, y.test1, beta.tar)
  get.auc(X.test2, y.test2, beta.tar)
  
  ###source only
  lam0 = fit0$lam.const
  w = list()
  TL = list()
  delta.all = list()
  X.all = y.all = NULL
  for(k in 1:K){
    w[[k]] <- as.numeric(ST.init(X.src[[k]], y.src[[k]])$beta0)
    print(paste0('w',k))
    p <- ncol(X.tar)
    w0 = as.numeric(w[[k]])
    new_X = w0[1]+X.tar%*%w0[-1]
    fit.tar <- cv.glmnet(x=cbind(new_X,X.tar), y=y.tar, nfolds=3, family='binomial')
    temp = coef(glmnet(x=cbind(new_X,X.tar), y=y.tar, lambda=fit.tar$lambda.min, family='binomial'))
    delta.all[[k]] <- (temp[2]-1)*w0 +temp[-2]
    TL[[k]] =  delta.all[[k]]+w0
    
    X.all = rbind(X.all, X.src[[k]])
    y.all = c(y.all, y.src[[k]])
  }
  
  ###calculate delta for each source + thresholding
  delta.TL = list()
  for(k in 1:K){
    delta.TL[[k]] = thres(delta.all[[k]], sqrt(n.tar), p) ###add threshold
  }
      
  ww = list()
  for(k in 1:K){
    ww[[k]] = thres(w[[k]], sqrt(n.src[k]), p) ###add threshold
  }
  
  ###SURE Screening
  auc.tar = c()
  for(k in 1:K){
    auc.tmp = get.auc(cbind(1, X.tar), y.tar, ww[[k]])
    auc.tar = c(auc.tar, auc.tmp)
  }
  K.1st = which(auc.tar==sort(auc.tar, decreasing = T)[1])
  K.2nd = which(auc.tar==sort(auc.tar, decreasing = T)[2])
  K.3rd = which(auc.tar==sort(auc.tar, decreasing = T)[3])
  K.4th = which(auc.tar==sort(auc.tar, decreasing = T)[4])
  
  ###singleTL
  beta.TL1 = TL[[K.1st]]
  beta.TL2 = TL[[K.2nd]]
  beta.TL3 = TL[[K.3rd]]
  beta.TL4 = TL[[K.4th]]
  print('singleTL done')
  
  ###create synthetic data
  data.syn <- create.synthetic(K, X.tar, n.src, r, B=list(TL[[1]], TL[[2]], TL[[3]], TL[[4]], TL[[5]], TL[[6]]))
  data.syn$X.syn[[1]] = X.src[[1]]
  data.syn$y.syn[[1]] = y.src[[1]]
  
  ###COMMUTE M=1
  if(K.1st==1){
    beta.syn12.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]]), delta=list(delta.TL[[1]], rep(0, p+1)))
    print('SynTL12 done')
    beta.syn123.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]]), delta=list(delta.TL[[1]], rep(0, p+1), rep(0, p+1)))
    print('SynTL123 done')
    beta.syn1234.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]],data.syn$X.syn[[K.4th]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]], data.syn$y.syn[[K.4th]]), delta=list(delta.TL[[1]], rep(0, p+1), rep(0, p+1), rep(0, p+1)))
    print('SynTL1234 done')
    rm(data.syn)
  }else if(K.2nd==1){
    beta.syn12.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]]), delta=list(rep(0, p+1), delta.TL[[2]]))
    print('SynTL12 done')
    beta.syn123.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]]), delta=list(rep(0, p+1), delta.TL[[2]], rep(0, p+1)))
    print('SynTL123 done')
    beta.syn1234.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]],data.syn$X.syn[[K.4th]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]], data.syn$y.syn[[K.4th]]), delta=list(rep(0, p+1), delta.TL[[2]], rep(0, p+1), rep(0, p+1)))
    print('SynTL1234 done')
    rm(data.syn)
  }else if(K.3rd==1){
    beta.syn12.M1 = ST.init(rbind(X.tar, data.syn$X.syn[[K.1st]], data.syn$X.syn[[K.2nd]]), c(y.tar,data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]]))$beta0
    print('SynTL12 done')
    beta.syn123.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]]), delta=list(rep(0, p+1), rep(0, p+1), delta.TL[[3]]))
    print('SynTL123 done')
    beta.syn1234.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]],data.syn$X.syn[[K.4th]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]], data.syn$y.syn[[K.4th]]), delta=list(rep(0, p+1), rep(0, p+1), delta.TL[[3]], rep(0, p+1)))
    print('SynTL1234 done')
    rm(data.syn)
  }else if(K.4th==1){
    beta.syn12.M1 = ST.init(rbind(X.tar, data.syn$X.syn[[K.1st]], data.syn$X.syn[[K.2nd]]), c(y.tar,data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]]))$beta0
    print('SynTL12 done')
    beta.syn123.M1 = ST.init(rbind(X.tar, data.syn$X.syn[[K.1st]], data.syn$X.syn[[K.2nd]], data.syn$X.syn[[K.3rd]]), c(y.tar,data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]]))$beta0
    print('SynTL12 done')
    beta.syn1234.M1 = Trans.global(X.tar, y.tar, X.src=list(data.syn$X.syn[[K.1st]],data.syn$X.syn[[K.2nd]],data.syn$X.syn[[K.3rd]],data.syn$X.syn[[K.4th]]), y.src=list(data.syn$y.syn[[K.1st]], data.syn$y.syn[[K.2nd]], data.syn$y.syn[[K.3rd]], data.syn$y.syn[[K.4th]]), delta=list(rep(0, p+1), rep(0, p+1), rep(0, p+1), delta.TL[[4]]))
    print('SynTL1234 done')
    rm(data.syn)
  }
  
  ###COMMUTE M=6 (PooledTL)
  beta.TL12 = Trans.global(X.tar, y.tar, X.src=list(X.src[[K.1st]],X.src[[K.2nd]]), y.src=list(y.src[[K.1st]], y.src[[K.2nd]]), delta=list(delta.TL[[K.1st]], delta.TL[[K.2nd]]))
  print('pooledTL12 done')
  beta.TL123 = Trans.global(X.tar, y.tar, X.src=list(X.src[[K.1st]],X.src[[K.2nd]],X.src[[K.3rd]]), y.src=list(y.src[[K.1st]], y.src[[K.2nd]], y.src[[K.3rd]]), delta=list(delta.TL[[K.1st]],delta.TL[[K.2nd]],delta.TL[[K.3rd]]))
  print('pooledTL123 done')
  #**will crash**beta.TL1234 = Trans.global(X.tar, y.tar, X.src=list(X.src[[K.1st]],X.src[[K.2nd]],X.src[[K.3rd]],X.src[[K.4th]]), y.src=list(y.src[[K.1st]], y.src[[K.2nd]], y.src[[K.3rd]], y.src[[K.4th]]), delta=list(delta.TL[[K.1st]],delta.TL[[K.2nd]],delta.TL[[K.3rd]],delta.TL[[K.4th]]))
  #print('pooledTL1234 done')
  
  #####################
  ### aggregation
  #####################
  
  ### COMMUTE with aggregation
  B.syn = cbind(beta.tar, beta.TL1, beta.TL2, beta.TL3, beta.TL4, beta.syn12.M1, beta.syn123.M1, beta.syn1234.M1)
  wt.syn.agg <- Agg.fun(B.syn, X.til, y.til)
  beta.syn.agg <- B.syn%*%wt.syn.agg
  
  wt.syn.agg.new <- Agg.fun.new(B.syn, X.til, y.til)
  beta.syn.agg.new <- B.syn%*%wt.syn.agg.new
  
  ### COMMUTE M=6 (pooledTL) with aggregation
  B.TL = cbind(beta.tar, beta.TL1, beta.TL2, beta.TL3, beta.TL4, beta.TL12, beta.TL123)#, beta.TL1234)
  wt.TL.agg <- Agg.fun(B.TL, X.til, y.til)
  beta.TL.agg <- B.TL%*%wt.TL.agg
  
  wt.TL.agg.new <- Agg.fun.new(B.TL, X.til, y.til)
  beta.TL.agg.new <- B.TL%*%wt.TL.agg.new
  
  ### Direct aggregation
  B.naive = matrix(0, nrow = p+1, ncol = K)
  for (k in 1:K) {
    B.naive[,k] = ww[[k]]
  }
  # top3 = c(K.1st, K.2nd, K.3rd)
  # for (k in 1:length(top3)) {
  #   B.naive[,k] = ww[[top3[k]]]
  # }
  B.naive = cbind(beta.tar, B.naive)
  wt.naive.agg <- Agg.fun(B.naive, X.til, y.til)
  beta.naive.agg <- B.naive%*%wt.naive.agg
  
  wt.naive.agg.new <- Agg.fun.new(B.naive, X.til, y.til)
  beta.naive.agg.new <- B.naive%*%wt.naive.agg.new

  
  ##########################
  #evaluation with test data
  ##########################
  w1 = ww[[1]]; w2 = ww[[2]]; w3 = ww[[3]]; w4 = ww[[4]]; w5 = ww[[5]]; w6 = ww[[6]]
  methods = c('beta.tar', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6',
              'beta.naive.agg','beta.syn.agg','beta.TL.agg',
              'beta.naive.agg.new','beta.syn.agg.new','beta.TL.agg.new')
  mseout = aucout = c()
  X.test = X.test1
  y.test = y.test1
  for(i in 1:length(methods)){
    auc_tmp = try(get.auc(X.test, y.test, get(methods[i])), silent=T)
    if(class(auc_tmp) == "try-error"){
      aucout = c(aucout, NA)
    }else{
      aucout = c(aucout, auc_tmp)
    }
  }
  print(cbind(methods, aucout))
  AUC1=cbind(methods, aucout)
  
  mseout = aucout = c()
  X.test = X.test2
  y.test = y.test2
  for(i in 1:length(methods)){
    auc_tmp = try(get.auc(X.test, y.test, get(methods[i])), silent=T)
    if(class(auc_tmp) == "try-error"){
      aucout = c(aucout, NA)
    }else{
      aucout = c(aucout, auc_tmp)
    }
  }
  print(cbind(methods, aucout))
  AUC2=cbind(methods, aucout)
  
  library(ggplot2)
  test = data.frame(methods=methods, aucout=aucout)
  test$group = c('beta.tar', rep('w',6), rep('Agg',3), rep('Agg new',3))
  ggplot(test, aes(x=methods, y=as.numeric(aucout), fill=methods)) +
    geom_bar(stat="identity", position=position_dodge()) +
    coord_cartesian(ylim=c(0.5, 0.85)) +
    facet_grid(cols = vars(group), scales = "free", space = "free")
  
  return(list(AUC1=AUC1, AUC2=AUC2))
}

set.seed(sim)
K = 6                                                                                        #total number of source population
X = data.matrix(ob_data[substring(ob_data$id, 1, 2)=='27' & ob_data$RACE==2, c(2:3, 7:2056)]) #Vanderbilt
y = data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='27' & ob_data$RACE==2, 'CASE_CONTROL_EXTREMEOBESITY']))
all = sample(1:nrow(X), 246) #300
agg = sample(all, 100)
test = all[!all%in%agg]
X.til = X[agg,]
y.til = y[agg]
X.tar = X[-all,]
y.tar = y[-all]

X.test1 <- cbind(1, X[test,])
y.test1 <- y[test]
X.test2 <- cbind(1, data.matrix(ob_data[substring(ob_data$id, 1, 2)=='52' & ob_data$RACE==2, c(2:3, 7:2056)]))    #Northwestern Black
y.test2 <- data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='52' & ob_data$RACE==2, 'CASE_CONTROL_EXTREMEOBESITY']))

X.src[[1]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='27' & ob_data$RACE!=2, c(2:3, 7:2056)]) #Vanderbilt University White
X.src[[2]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='16' & ob_data$RACE!=2, c(2:3, 7:2056)]) #Marshfield Clinic White
X.src[[3]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='49' & ob_data$RACE!=2, c(2:3, 7:2056)]) #Mayo Clinic White
X.src[[4]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='63' & ob_data$RACE!=2, c(2:3, 7:2056)]) #Geisinger White
X.src[[5]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='52' & ob_data$RACE!=2, c(2:3, 7:2056)]) #Northwestern White
X.src[[6]] =  data.matrix(ob_data[substring(ob_data$id, 1, 2)=='74' & ob_data$RACE==2, c(2:3, 7:2056)]) #Mt Sinai Black

y.src[[1]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='27' & ob_data$RACE!=2, 'CASE_CONTROL_EXTREMEOBESITY']))
y.src[[2]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='16' & ob_data$RACE!=2, 'CASE_CONTROL_EXTREMEOBESITY']))
y.src[[3]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='49' & ob_data$RACE!=2, 'CASE_CONTROL_EXTREMEOBESITY']))
y.src[[4]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='63' & ob_data$RACE!=2, 'CASE_CONTROL_EXTREMEOBESITY']))
y.src[[5]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='52' & ob_data$RACE!=2, 'CASE_CONTROL_EXTREMEOBESITY']))
y.src[[6]] =  data.matrix(as.numeric(ob_data[substring(ob_data$id, 1, 2)=='74' & ob_data$RACE==2, 'CASE_CONTROL_EXTREMEOBESITY']))

p = ncol(X.tar)                                                                   #dimension of p
n0 = nrow(X.tar)                                                                  #target population size
nt = c(nrow(X.src[[1]]), nrow(X.src[[2]]), nrow(X.src[[3]]), nrow(X.src[[4]]), nrow(X.src[[5]]), nrow(X.src[[6]]))    #source population size
r = 25                                                                       #synthetic data size = r*nt


results = simu(X.tar, y.tar, X.src, y.src, p, K, n0, nt, r, X.til, y.til, X.test1, y.test1, X.test2, y.test2)
save(results, file=paste0('/n/home02/tiangu/JAMIA/JBI/results_emerge/sim_', sim, '.RData'))


# site key: 
#1: Vanderbilt University=27; 
#2: Marshfield Clinic=16    
#3: Mayo Clinic=49; 
#4: Northwestern University=52; 
#5: Geisinger=63; 
#6: Mt Sinai=74; 
#Cincinnati Childrens=81; Meharry=88; CHOP=95; Harvard=68; Kaiser Permanente/UW=38; Columbia=42; Boston Childrens=23; 

####### correlation plot
# library(dplyr)
# library(ggcorrplot)
# cor.tar <- cor(X[,6:96])  
# cor1 <- cor(X.src[[1]][,6:96])   
# cor2 <- cor(X.src[[2]][,6:96])   
# cor3 <- cor(X.src[[3]][,6:96])   
# cor4 <- cor(X.src[[4]][,6:96])   
# cor5 <- cor(X.src[[5]][,6:96])   
# cor6 <- cor(X.src[[6]][,6:96])   
# cor_combined <- cor.tar
# cor_combined[upper.tri(cor_combined)] <- cor1[upper.tri(cor1)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 
# 
# cor_combined[upper.tri(cor_combined)] <- cor2[upper.tri(cor2)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 
# 
# cor_combined[upper.tri(cor_combined)] <- cor3[upper.tri(cor3)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 
# 
# cor_combined[upper.tri(cor_combined)] <- cor4[upper.tri(cor4)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 
# 
# cor_combined[upper.tri(cor_combined)] <- cor5[upper.tri(cor5)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 
# 
# cor_combined[upper.tri(cor_combined)] <- cor6[upper.tri(cor6)]
# ggcorrplot(cor_combined, tl.col = "red", outline.col = "white") 



#########################
######### proccess data
#########################
library(ggplot2)

#total = c(45,  125, 126, 145, 150, 175, 191, 215, 217, 237) #TL.agg-syn.agg<0.08
total = c(1, 5, 11, 13, 14, 20, 24, 30, 31, 41, 35, 49) 
tar=w1=w2=w3=w4=w5=w6=TL.agg=syn.agg=naive.agg=matrix(NA, length(total),1)
DNN = read.csv('/n/home02/tiangu/JAMIA/JBI/pooled_results/DNN_emerge_4layers_test1.csv', header=F)$V1
SER = read.csv('/n/home02/tiangu/JAMIA/JBI/pooled_results/SER_emerge_AUC1.csv', header=F)$V1

metric = 'aucout'
for(i in 1:length(total)){
  print(i)
  import = try(load(paste0('/n/home02/tiangu/JAMIA/JBI/results_emerge/sim_', total[i], '.RData')))
  if(inherits(import, 'try-error')){
    next
  }
  results$AUC = results$AUC1

  tar[i] = as.numeric(results$AUC[1, metric])
  w1[i] = as.numeric(results$AUC[2, metric])
  w2[i] = as.numeric(results$AUC[3, metric])
  w3[i] = as.numeric(results$AUC[4, metric])
  w4[i] = as.numeric(results$AUC[5, metric])
  w5[i] = as.numeric(results$AUC[6, metric])
  w6[i] = as.numeric(results$AUC[7, metric])
  naive.agg[i] = as.numeric(results$AUC[8, metric])
  syn.agg[i] = as.numeric(results$AUC[9, metric])
  TL.agg[i] = as.numeric(results$AUC[10, metric])
  # naive.agg[i] = as.numeric(results$AUC[11, metric])
  # syn.agg[i] = as.numeric(results$AUC[12, metric])
  # TL.agg[i] = as.numeric(results$AUC[13, metric])
}


########### bar plot
dat = data.frame(AUC=c(mean(tar, na.rm=T), mean(w1, na.rm=T), mean(w2, na.rm=T), mean(w3, na.rm=T), mean(w4, na.rm=T), mean(w5, na.rm=T), mean(w6, na.rm=T),
                       mean(DNN, na.rm=T), mean(SER, na.rm=T), mean(naive.agg, na.rm=T),
                       mean(syn.agg, na.rm=T), mean(TL.agg, na.rm=T)),
                 Q1=c(quantile(tar, .25, na.rm=T), quantile(w1, .25, na.rm=T), quantile(w2, .25, na.rm=T), quantile(w3, .25, na.rm=T), quantile(w4, .25, na.rm=T), quantile(w5, .25, na.rm=T), quantile(w6, .25, na.rm=T),
                      quantile(DNN, .25, na.rm=T), quantile(SER, .25, na.rm=T), quantile(naive.agg, .25, na.rm=T), 
                      quantile(syn.agg, .25, na.rm=T), quantile(TL.agg, .25, na.rm=T)),
                 Q3=c(quantile(tar, .75, na.rm=T), quantile(w1, .75, na.rm=T), quantile(w2, .75, na.rm=T), quantile(w3, .75, na.rm=T), quantile(w4, .75, na.rm=T), quantile(w5, .75, na.rm=T), quantile(w6, .75, na.rm=T),
                      quantile(DNN, .75, na.rm=T), quantile(SER, .75, na.rm=T), quantile(naive.agg, .75, na.rm=T), 
                      quantile(syn.agg, .75, na.rm=T), quantile(TL.agg, .75, na.rm=T)),
                 Methods=c('Vanderbilt AA', 'Vanderbilt White', 'Marshfield White', 'Mayo Clinic White', 'Northwestern White', 'Geisinger White', 'Mount Sinai AA','DNN','SER','Direct','M=1','M=6'))
dat$Methods = factor(dat$Methods, levels = c('Vanderbilt AA', 'Vanderbilt White', 'Marshfield White', 'Mayo Clinic White', 'Northwestern White', 'Geisinger White','Mount Sinai AA','DNN','SER','Direct','M=1','M=6'))
dat$group = c(rep('Target', 1), rep('Source', 6), rep('Others', 3), rep('COMMUTE', 2))
dat$group = factor(dat$group, levels = c('Target', 'Source', 'Others','COMMUTE'))

dat$vjust = c(-2.5, -1.5, -1.5, -2, -1.9, -1, -1.5, -1.5, -1.5, -1.5, -2.5, -1.5)
p1 = ggplot(dat, aes(x=Methods, y=AUC, fill=Methods)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=Q1, ymax=Q3), width=.2, position=position_dodge(.9)) +
  ylab("AUC") + xlab("") +
  coord_cartesian(ylim=c(0.5, 1)) +
  #theme_bw() +
  ggtitle('Test on Vanderbilt AA') +
  geom_text(aes(label = sprintf("%.2f", AUC), y=AUC, vjust=vjust)) +
  theme(text = element_text(size=17),
        legend.key.size = unit(0.8, 'cm'),
        legend.text=element_text(size=14),
        legend.position = "none") +
  theme(axis.text.x=element_text(angle=15,hjust=1)) +
  facet_grid(cols = vars(group), scales = "free", space = "free")


################
## External test
#################
DNN = read.csv('/n/home02/tiangu/JAMIA/JBI/pooled_results/DNN_emerge_4layers_test2.csv', header=F)$V1
SER = read.csv('/n/home02/tiangu/JAMIA/JBI/pooled_results/SER_emerge_AUC2.csv', header=F)$V1
metric = 'aucout'
for(i in 1:length(total)){
  print(i)
  import = try(load(paste0('/n/home02/tiangu/JAMIA/JBI/results_emerge/sim_', total[i], '.RData')))
  if(inherits(import, 'try-error')){
    next
  }
  results$AUC = results$AUC2
  
  tar[i] = as.numeric(results$AUC[1, metric])
  w1[i] = as.numeric(results$AUC[2, metric])
  w2[i] = as.numeric(results$AUC[3, metric])
  w3[i] = as.numeric(results$AUC[4, metric])
  w4[i] = as.numeric(results$AUC[5, metric])
  w5[i] = as.numeric(results$AUC[6, metric])
  w6[i] = as.numeric(results$AUC[7, metric])
  naive.agg[i] = as.numeric(results$AUC[8, metric])
  syn.agg[i] = as.numeric(results$AUC[9, metric])
  TL.agg[i] = as.numeric(results$AUC[10, metric])
  # naive.agg[i] = as.numeric(results$AUC[11, metric])
  # syn.agg[i] = as.numeric(results$AUC[12, metric])
  # TL.agg[i] = as.numeric(results$AUC[13, metric])
}


dat = data.frame(AUC=c(mean(tar, na.rm=T), mean(w1, na.rm=T), mean(w2, na.rm=T), mean(w3, na.rm=T), mean(w4, na.rm=T), mean(w5, na.rm=T), mean(w6, na.rm=T),
                       mean(DNN, na.rm=T), mean(SER, na.rm=T), mean(naive.agg, na.rm=T),
                       mean(syn.agg, na.rm=T), mean(TL.agg, na.rm=T)),
                 Q1=c(quantile(tar, .25, na.rm=T), quantile(w1, .25, na.rm=T), quantile(w2, .25, na.rm=T), quantile(w3, .25, na.rm=T), quantile(w4, .25, na.rm=T), quantile(w5, .25, na.rm=T), quantile(w6, .25, na.rm=T),
                      quantile(DNN, .25, na.rm=T), quantile(SER, .25, na.rm=T), quantile(naive.agg, .25, na.rm=T), 
                      quantile(syn.agg, .25, na.rm=T), quantile(TL.agg, .25, na.rm=T)),
                 Q3=c(quantile(tar, .75, na.rm=T), quantile(w1, .75, na.rm=T), quantile(w2, .75, na.rm=T), quantile(w3, .75, na.rm=T), quantile(w4, .75, na.rm=T), quantile(w5, .75, na.rm=T), quantile(w6, .75, na.rm=T),
                      quantile(DNN, .75, na.rm=T), quantile(SER, .75, na.rm=T), quantile(naive.agg, .75, na.rm=T), 
                      quantile(syn.agg, .75, na.rm=T), quantile(TL.agg, .75, na.rm=T)),
                 Methods=c('Vanderbilt AA', 'Vanderbilt White', 'Marshfield White', 'Mayo Clinic White', 'Northwestern White', 'Geisinger White', 'Mount Sinai AA','DNN','SER','Direct','M=1','M=6'))
dat$Methods = factor(dat$Methods, levels = c('Vanderbilt AA', 'Vanderbilt White', 'Marshfield White', 'Mayo Clinic White', 'Northwestern White', 'Geisinger White','Mount Sinai AA','DNN','SER','Direct','M=1','M=6'))
dat$group = c(rep('Target', 1), rep('Source', 6), rep('Others', 3), rep('COMMUTE', 2))
dat$group = factor(dat$group, levels = c('Target', 'Source', 'Others','COMMUTE'))
dat$vjust = c(-2.5, -2, -1.5, -1, -1, -0.5, -1, -1.5, -1.5, -1.5, -1.5, -1.5)

p2 = ggplot(dat, aes(x=Methods, y=AUC, fill=Methods)) +
  geom_bar(stat="identity", position=position_dodge()) +
  geom_errorbar(aes(ymin=Q1, ymax=Q3), width=.2, position=position_dodge(.9)) +
  ylab("AUC") + xlab("") +
  coord_cartesian(ylim=c(0.5, 1)) +
  #theme_bw() +
  ggtitle('Test on Northwestern AA') +
  geom_text(aes(label = sprintf("%.2f", AUC), y=AUC, vjust=vjust)) +
  theme(text = element_text(size=17),
        legend.key.size = unit(0.8, 'cm'),
        legend.text=element_text(size=14),
        legend.position = "none") +
  theme(axis.text.x=element_text(angle=15,hjust=1)) +
  facet_grid(cols = vars(group), scales = "free", space = "free")


library(patchwork)
p1/p2
# 
# ggsave(path = '/n/home02/tiangu/JAMIA/JBI', width = 10, height = 8, filename='figure4.png', dpi=200)



