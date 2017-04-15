library(h2o)
library(pogs)
library(glmnet)
library(data.table)
#library(feather)
library(MatrixModels)

pogs  <-TRUE
glmnet<-TRUE
h2o   <-FALSE
alpha <- 1
family <- "gaussian"
intercept <- TRUE

f <- "~/ipums_2000-2015.csv"
response <- "INCEARN"

file <- paste0("train.csv")
if (FALSE) {
  source("ipums_prep.R")
} else {
  DT <- fread(file)
}

cols <- setdiff(names(DT),c(response))

n <- nrow(DT)
p <- ncol(DT)
n
p

set.seed(1234)
train_rows <- 1:(floor(0.8*n)+1) #sample(1:n, .8*n)

train <- DT[train_rows,]
valid <- DT[-train_rows,]

dim(train)
dim(valid)

train_x  <- as.matrix((train[,cols,with=FALSE]))
dim(train_x)

train_y  <- as.numeric(train[[response]])
length(train_y)

valid_x  <- as.matrix((valid[,cols,with=FALSE]))
dim(valid_x)

valid_y  <- as.numeric(valid[[response]])
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
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=NULL, cutoff=FALSE, intercept=intercept, #lambda.min.ratio=1e-6,
                 ,params=list(rel_tol=1e-4, abs_tol=1e-5, rho=1, max_iter=10000, adaptive_rho=FALSE, equil=FALSE, wDev=0L, verbose=4)
  )
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")
  
  print("POGS GPU: ")
  print(e1-s1)
  
  pogsbeta <- score(pogs, pogs_pred_y, valid_y)
}

## GLMNET
if (glmnet) {
#  library(foreach)
#  library(doMC)
#  registerDoMC(16)
#  foreach(a=1:16) %dopar% {
#    alpha = (a-1)/15
    alpha = alpha

    if (family=="binomial") {
      y = as.factor(train_y)
    } else {
      y = train_y
    }
    s2 <- proc.time()
    glmnet = glmnet(x = train_x, y = y, family = family, alpha = alpha, standardize=FALSE, intercept=intercept)
    e2 <- proc.time()
    glmnet_pred_y = predict(glmnet, valid_x, type="response")
    
    print("GLMNET CPU")
    print(e2-s2)
    print(paste0("alpha:", alpha))
    glmnetbeta <- score(glmnet, glmnet_pred_y, valid_y)
    print(glmnet)
#  }
}


## H2O 
## SEE ABOVE FOR MODEL ON DATA CREATION
if (h2o) {
  h2o.init(nthreads=-1)
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

quit()

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
    
    ## INACCURATE
    #model <- pogsnet(x = t(L), y = z, family = family, alpha = a, lambda=NULL, cutoff=FALSE
    #                         #,params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1, max_iter=10000, adaptive_rho=FALSE, equil=TRUE)
    #)
    
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





