library(h2o)
library(pogs)
library(glmnet)
library(data.table)
#library(feather)
library(MatrixModels)
h2o.init(nthreads=-1)

pogs  <-FALSE
glmnet<-TRUE
h2o   <-FALSE
alpha <- .5
family <- "gaussian"

f <- "~/ipums_2000-2015_head4M.csv"
response <- "INCEARN"

file <- paste0("/tmp/train.csv")
if (FALSE) {
  DT <- fread(f, nrows=100000)
  DT = DT[INCEARN>100] ## only keep rows with valid response (0 is ok)
  #DT = DT[INCTOT!=9999999]
  DT[,c('HHINCOME','INCWAGE','INCTOT','FTOTINC',"INCBUS00",'INCSS',"INCWELFR","INCINVST", "INCRETIR", "INCSUPP", "INCOTHER"):=NULL] ## drop highly correlated columns
  ncol(DT)
  #DT[,CLUSTER:=as.factor(as.numeric(CLUSTER %% (max(CLUSTER)-min(CLUSTER)+1)))]
  DT[,CLUSTER:=NULL] ## has > 30k factors
  
  ## label-encoding of categoricals
  feature.names <- setdiff(names(DT), response)
  sum <- 0
  for (ff in feature.names) {
    tt <- uniqueN(DT[[ff]])
    if (tt < 1000 && tt > 1) {
      DT[, (ff):=factor(DT[[ff]])]  
      print(paste0(ff,"has ",tt," levels"))
      sum <- sum + tt
    }
    if (tt < 2) {
      print(paste0("dropping constant column: ", ff))
      DT[, (ff):=NULL]
    }
  }
  print(sum)
  ncol(DT)
  DT
  
  numCols <- names(DT)[which(sapply(DT, is.numeric))]
  catCols <- names(DT)[which(sapply(DT, is.factor))]
  numCols
  catCols
  
  
  ## impute missing values, drop near-const cols and standardize the data
  cols <- setdiff(numCols,c(response))
  for (c in cols) {
    DT[!is.finite(DT[[c]]), (c):=mean(DT[[c]], na.rm=TRUE)]
    if (!is.finite(sd(DT[[c]])) || sd(DT[[c]])<1e-4) 
      DT[,(c):=NULL]
    else
      DT[,(c):=scale(as.numeric(DT[[c]]))]
  }
  ncol(DT)
  
  ## one-hot encode the categoricals, but keep everything dense
  DT2 <- as.data.table(model.matrix(DT[[response]]~., data = DT[,c(catCols), with=FALSE], sparse=FALSE))
  ncol(DT2)
  
  DT2[,(numCols):=DT[,c(numCols), with=FALSE]]
  ncol(DT2)
  
  DT <- DT2
  ncol(DT)
  
  ## all cols are now numeric
  all(sapply(DT, is.numeric))

  ## check validity of data
  all(!is.na(DT))
  all(sapply(DT, function(x) all(is.finite(x))))
  
  ## TODO: vtreat
  fwrite(DT, file)
  
} else {
  DT <- fread(file)
}

cols <- setdiff(names(DT),c(response))

n <- nrow(DT)
p <- ncol(DT)
n
p

set.seed(1234)
train_rows <- sample(1:n, .8*n)

train <- DT[train_rows,]
valid <- DT[-train_rows,]

dim(train)
dim(valid)

train_x  <- as.matrix((train[,cols,with=FALSE]))
dim(train_x)

train_y  <- train[[response]]
length(train_y)

valid_x  <- as.matrix((valid[,cols,with=FALSE]))
dim(valid_x)

valid_y  <- valid[[response]]
length(valid_y)

score <- function(model, preds, actual) {
  rmse = apply(preds, 2, function(x) sqrt(mean((x-actual)^2)))
  r = cbind(lambda=log10((model$lambda)), rmse)
  plot(r)
  print(paste0("RMSE: ", min(rmse)))
  idx = which((rmse)==min(rmse))
  print(paste0("best lambda: ", idx, "-th largest"))
  print(paste0("lambda: ", model$lambda)[idx])
  print(paste0("dofs: ", model$df)[idx])
  print(paste0("number of coefs: ", length(which(model$beta[,idx]!=0))))
  #print(model$beta[which(model$beta[,idx]!=0),idx])
  plot(model, xvar="lambda")
  coef(model, s=model$lambda[idx])
}

## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=NULL, cutoff=FALSE,
                 params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1,
                             max_iter=20000, 
                             adaptive_rho=TRUE, equil=TRUE, wDev=0L))
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")
  
  print("POGS GPU: ")
  print(e1-s1)
  
  pogsbeta <- score(pogs, pogs_pred_y, valid_y)
}

## GLMNET
if (glmnet) {
  if (family=="binomial") {
    y = as.factor(train_y)
  } else {
    y = train_y
  }
  s2 <- proc.time()
  glmnet = glmnet(x = train_x, y = y, family = family, alpha = alpha)
  e2 <- proc.time()
  glmnet_pred_y = predict(glmnet, valid_x, type="response")
  
  print("GLMNET CPU")
  print(e2-s2)
  
  glmnetbeta <- score(glmnet, glmnet_pred_y, valid_y)
}


## H2O 
## SEE ABOVE FOR MODEL ON DATA CREATION
if (h2o) {
  ## Quick H2O model
  if (FALSE) {
    h2oglm <- h2o.glm(alpha=alpha,training_frame=as.h2o(train),y=response,lambda_search = TRUE)
    h2oglm
    print("H2O CPU")
    print(h2o.rmse(h2o.performance(h2oglm,newdata=as.h2o(valid))))
    
    print("StdDev of Response")
    print(sd(valid[[response]]))
  } else {
    df.hex <- h2o.importFile(file)
    n<-nrow(df.hex)
    set.seed(1234)
    train_rows <- sample(1:n, .8*n)
    
    train.hex <- df.hex[sort(train_rows),]
    valid.hex <- df.hex[-sort(train_rows),]
    
    if (family=="binomial") {
      train.hex[[response]] <- as.factor(train.hex[[response]])
      valid.hex[[response]] <- as.factor(valid.hex[[response]])
    }
    
    s3 <- proc.time()
    h2omodel <- h2o.glm(x=names(train.hex), y=response, training_frame=train.hex, family=family, alpha = alpha, lambda_search=TRUE, solver="COORDINATE_DESCENT_NAIVE")
    e3 <- proc.time()
    h2opreds <- h2o.predict(h2omodel, valid.hex)
    summary(h2opreds)
    
    print("H2O CPU ")
    print(e3-s3)
    
    regpath = h2o.getGLMFullRegularizationPath(h2omodel)
    n = dim(regpath$coefficients)[1]
    coefs = NULL
    for (i in 1:n) {
      coefs = regpath$coefficients[i,]
      h2omodel2 <- h2o.makeGLMModel(h2omodel,coefs)
      h2opreds <- h2o.predict(h2omodel2, valid.hex)
      if (family == "gaussian") {
        rmse[i] <- h2o.rmse(h2o.make_metrics(h2opreds[,1], valid.hex[[response]]))
      } else {
        #h2o.auc(h2o.make_metrics(h2opreds[,3], valid.hex[[response]])))
      }
    }
    plot(rmse)
    print(min(rmse))
  }
}


print("FAST SOLVER IN PROJECTED SPACE")

## Proposed solution for fast GPU solver
## Idea: Build the model in the projected space

## One-time projection of training data
AtA <- (t(train_x) %*% train_x)
Atb <- (t(train_x) %*% train_y)
## One-time projection of validation data
validAtA <- t(valid_x) %*% valid_x
validAtb <- t(valid_x) %*% valid_y

## From now on, do all these on tiny data (still GPU, do a lot of things at once)
L <- t(chol(AtA+1e-5*diag(nrow(AtA))))         ## get L of cholesky factorization: L Lt = At A
z <- forwardsolve(L, Atb) ## solve L z = At b

system.time(
  for (al in (alpha*100+1)) {
    #  for (al in seq(1,101,10)) {
    a <- 0.01*(al-1)
    print(paste0("alpha: ", a))
    
    ## FAST - always a small gram matrix given.
    model <- glmnet(x=t(L), y=z, family=family, alpha=a)
    
    ## SLOW -- too small data?
    # model <- pogsnet(x = train_x, y = train_y, family = family, alpha = a, lambda=NULL, cutoff=FALSE,
    #                         params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1,
    #                                     max_iter=10000,
    #                                     adaptive_rho=FALSE, equil=TRUE))
    
    ## Option 1) Benchmark: Score normally on full validation set using the "regular" coefficients
    p <- predict(model, valid_x, type="response")
    modelbeta <- score(model, p, valid_y)
    
    ## Option 2) Faster: Can even find the lowest RMSE of the model in the projected space!
    ## NOT AS ACCURATE?!
    Atp <- as.matrix(validAtA %*% model$beta) + model$a0         ## projected prediction
    rmse = apply(Atp, 2, function(x) sqrt(mean((x-validAtb)^2))) ## projected rmse 
    r = cbind(lambda=log10((model$lambda)), rmse)
    plot(r)
    print(paste0("RMSE: ", min(rmse)))
    idx = which((rmse)==min(rmse))
    print(paste0("best lambda: ", idx, "-th largest"))
    #plot(rmse)
    print("")
  }
)




