library(h2o)
library(pogs)
library(glmnet)
library(data.table)
h2o.init(nthreads=-1)

pogs  <-TRUE
glmnet<-TRUE
h2o   <-TRUE
alpha <- 0.8
family <- "gaussian"
#family <- "binomial"

 
if (FALSE) {
  #https://www.kaggle.com/c/springleaf-marketing-response/data
  N<-145231 ## max
  N<-10000  ## ok for accuracy tests
  H <- round(0.8*N) ## need to split into train/test since kaggle test set has no labels
  #f <- "gunzip -c ../data/springleaf/train.csv.zip"
  f <- "~/kaggle/springleaf/input/train.csv"
  response <- 'target'

  file <- paste0("/tmp/train.",N,".csv")
  if (FALSE) {
    ## DATA PREP
    DT <- fread(f, nrows=N)
    
    DT[['ID']] <- NULL ## ignore ID
    
    ## label encoding
    #feature.names <- setdiff(names(DT), response)
    #for (f in feature.names) {
    #  if (class(DT[[f]])=="character") {
    #    levels <- unique(c(DT[[f]]))
    #    DT[[f]] <- as.integer(factor(DT[[f]], levels=levels))
    #  }
    #}
    DT[is.na(DT)] <- 0
    DT <- DT[,sapply(DT, is.numeric), with=FALSE]
    DT <- DT[,apply(DT, 2, var, na.rm=TRUE) > 0, with=FALSE] ## drop const cols
    #  cols <- names(DT)
    cols <- setdiff(names(DT),c(response))
    for (c in cols) {
      DT[,(c):=scale(DT[[c]])]
    }
    DT <- DT[,apply(DT, 2, var, na.rm=TRUE) > 0.1, with=FALSE] ## drop near-const cols
    fwrite(DT, file)
  } else {
    DT <- fread(file)
    cols <- setdiff(names(DT),c(response))
  }
  
  train <- DT[1:H,]
  valid <- DT[(H+1):N,]
  
  train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
  train_y  <- as.numeric(as.vector(train[[response]]))
  valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
  valid_y  <- as.numeric(as.vector(valid[[response]]))
} else {
  
  # # Generate synthetic data
  # set.seed(19875)  # Set seed for reproducibility
  # n <- 10000  # Number of observations
  # p <- 1000  # Number of predictors included in model
  # real_p <- 100  # Number of true predictors
  # x <- matrix(rnorm(n*p), nrow=n, ncol=p)
  # y <- apply(x[,1:real_p], 1, sum) + rnorm(n)
  # 
  df <- iris[,-5]
  cuts <- apply(df[,c(2,3)], 2, cut, c(-Inf,seq(0, 10, 0.1), Inf))
  cuts <- cbind(as.data.frame(cuts), df[,c(1,4)])
  df <- data.frame(model.matrix(~.-1,cuts))
  n <- nrow(df)
  p <- ncol(df)
  
  # Split data into train and validation
  train_rows <- sample(1:n, .8*n)
  const_cols <- which(apply(df[train_rows,], 2, sd) ==0)
  df <- df[,-const_cols]
  df <- df + 0.01* matrix(rnorm(n*p), nrow=n, ncol=p) ## optional: add noise
  
  summary(df)
  response <- "Petal.Width"
  
  x <- as.matrix(df[,setdiff(names(df),response)])
  y <- as.matrix(df[[response]])
  train_x <- x[train_rows, ]
  valid_x <- x[-train_rows, ]
  train_y <- y[train_rows]
  valid_y <- y[-train_rows]
}


score <- function(model, preds, actual) {
  rmse = apply(preds, 2, function(x) sqrt(mean((x-actual)^2)))
  r = cbind(lambda=log10(rev(model$lambda)), rmse)
  plot(r)
  print(paste0("RMSE: ", min(rmse)))
  idx = which(rev(rmse)==min(rmse))
  print(paste0("best lambda: ", idx, "-th largest"))
  print(paste0("lambda: ", model$lambda)[idx])
  print(paste0("dofs: ", model$df)[idx])
  plot(model, xvar="lambda")
}

## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=NULL, cutoff=FALSE,
                 params=list(rel_tol=1e-3, abs_tol=1e-3, rho=1,
                   #max_iter=200, 
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

  h2oglm <- h2o.glm(alpha=alpha,training_frame=as.h2o(df[train_rows,]),validation_frame=as.h2o(df[-train_rows,]),y=response,lambda_search = TRUE)
  h2oglm
  print("H2O CPU")
  print(h2o.rmse(h2o.performance(h2oglm,valid=TRUE)))
  
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
L <- t(chol(AtA))         ## get L of cholesky factorization: L Lt = At A
z <- forwardsolve(L, Atb) ## solve L z = At b

## Solve Lt x = z with MSE loss and all L1/L2 regularization terms
model <- glmnet(x=t(L), y=z, family=family, alpha=alpha) ## TODO: can cheaply add CV, alpha search etc.

## Validation

## Option 1) Benchmark: Score normally on full validation set using the "regular" coefficients
p <- predict(model, valid_x, type="response")
score(model, p, valid_y)

## Option 2) Faster: Can even find the lowest RMSE of the model in the projected space!
Atp <- as.matrix(validAtA %*% model$beta) + model$a0         ## projected prediction
rmse = apply(Atp, 2, function(x) sqrt(mean((x-validAtb)^2))) ## projected rmse 
idx = which(rev(rmse)==min(rmse))
print(paste0("best lambda: ", idx, "-th largest"))

