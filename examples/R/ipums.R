library(h2o)
library(pogs)
library(glmnet)
library(data.table)
h2o.init(nthreads=-1)

pogs  <-TRUE
glmnet<-TRUE
h2o   <-TRUE
alpha <- .5
family <- "gaussian"
#family <- "binomial"


#https://www.kaggle.com/c/springleaf-marketing-response/data
#N<-145231 ## max
#f <- "gunzip -c ../data/springleaf/train.csv.zip"
#f <- "~/kaggle/springleaf/input/train.csv"
#response <- 'target'
f <- "~/ipums_2000-2015_head4M.csv"
response <- "INCEARN"
#f <- "~/h2o-3/smalldata/junit/weather.csv"
#response <- 'Pressure3pm'

file <- paste0("/tmp/train.csv")
if (TRUE) {
  DT <- fread(f, nrows=10000, colClasses = "double")
  DT[,c('HHINCOME','INCWAGE'):=NULL]

  ## label encoding
  #feature.names <- setdiff(names(DT), response)
  #for (f in feature.names) {
  #  if (class(DT[[f]])=="character") {
  #    levels <- unique(c(DT[[f]]))
  #    DT[[f]] <- as.integer(factor(DT[[f]], levels=levels))
  #  }
  #}
  
  
  ## only keep numeric columns
  DT[,which(!sapply(DT, is.numeric)):=NULL]
  
  ## impute missing values and drop near-const cols
  cols <- setdiff(names(DT),c(response))
  for (c in cols) {
    DT[is.na(DT[[c]]), (c):=mean(DT[[c]], na.rm=TRUE)]
    if (is.na(sd(DT[[c]])) || sd(DT[[c]])<1e-4) 
      DT[,(c):=NULL]
  }
  
  ## standardize the data (and add a little noise)
  cols <- setdiff(names(DT),c(response))
  for (c in cols) {
    DT[,(c):=scale(DT[[c]]) + 0.01*rnorm(nrow(DT))]
  }
  
  fwrite(DT, file)
} else {
  DT <- fread(file)
  cols <- setdiff(names(DT),c(response))
}
n <- nrow(DT)
p <- ncol(DT)

train_rows <- sample(1:n, .8*n)

train <- DT[train_rows,]
valid <- DT[-train_rows,]

train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
train_y  <- as.numeric(as.vector(train[[response]]))
valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
valid_y  <- as.numeric(as.vector(valid[[response]]))


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
  #print(which(model$beta[,idx]!=0))
  plot(model, xvar="lambda")
}

## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=NULL, cutoff=FALSE,
                 params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1,
                             max_iter=20000, 
                             adaptive_rho=TRUE, equil=TRUE))
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")
  
  print("POGS GPU: ")
  print(e1-s1)
  
  score(pogs, pogs_pred_y, valid_y)
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
  
  score(glmnet, glmnet_pred_y, valid_y)
}


## H2O 
## SEE ABOVE FOR MODEL ON DATA CREATION
if (h2o) {
  
  
  ## Quick H2O model
  
  h2oglm <- h2o.glm(alpha=alpha,training_frame=as.h2o(train),y=response,lambda_search = TRUE)
  h2oglm
  print("H2O CPU")
  print(h2o.rmse(h2o.performance(h2oglm,newdata=as.h2o(valid))))
  
  print("StdDev of Response")
  print(sd(valid[[response]]))
  
  if (FALSE) {
    df.hex <- h2o.importFile(file)
    train.hex <- df.hex[1:H,]
    valid.hex <- df.hex[(H+1):N,]
    
    if (family=="binomial") {
      train.hex[[response]] <- as.factor(train.hex[[response]])
      valid.hex[[response]] <- as.factor(valid.hex[[response]])
    }
    
    s3 <- proc.time()
    h2omodel <- h2o.glm(x=cols, y=response, training_frame=train.hex, family=family, alpha = alpha, lambda_search=TRUE, solver="COORDINATE_DESCENT_NAIVE")
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
Lt <- chol(AtA)         ## get Lt of cholesky factorization: L Lt = At A
z <- forwardsolve(t(Lt), Atb) ## solve L z = At b

## Solve Lt x = z with MSE loss and all L1/L2 regularization terms, lambda path, everything.

system.time(
#  for (al in (alpha*100+1)) {
  for (al in seq(1,101,10)) {
    a <- 0.01*(al-1)
    print(paste0("alpha: ", a))
    
    ## FAST - always a small gram matrix given.
    model <- glmnet(x=Lt, y=z, family=family, alpha=a)

    ## SLOW -- too small data?
    # model <- pogsnet(x = train_x, y = train_y, family = family, alpha = a, lambda=NULL, cutoff=FALSE,
    #                         params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1,
    #                                     max_iter=10000,
    #                                     adaptive_rho=FALSE, equil=TRUE))

    ## Option 1) Benchmark: Score normally on full validation set using the "regular" coefficients
    p <- predict(model, valid_x, type="response")
    score(model, p, valid_y)
   
    ## Option 2) Faster: Can even find the lowest RMSE of the model in the projected space!
    ## NOT AS ACCURATE?!
#    Atp <- as.matrix(validAtA %*% model$beta) + model$a0         ## projected prediction
#    rmse = apply(Atp, 2, function(x) sqrt(mean((x-validAtb)^2))) ## projected rmse 
#    idx = which(rmse==min(rmse))
#    print(paste0("best lambda: ", idx, "-th largest"))
#    plot(rmse)
    print("")
  }
)




