library(h2o)
library(pogs)
library(glmnet)
library(data.table)

#https://www.kaggle.com/c/springleaf-marketing-response/data
N<-145231 ## max
#N<-10000  ## ok for accuracy tests
H <- round(0.8*N) ## need to split into train/test since kaggle test set has no labels
#f <- "gunzip -c ../data/springleaf/train.csv.zip"
f <- "~/kaggle/springleaf/input/train.csv"

response <- 'target'
family <- "gaussian"
#family <- "binomial"
pogs  <-TRUE
glmnet<-FALSE
h2o   <-FALSE
alpha <- 0.5

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
  DT <- DT[,apply(DT, 2, var, na.rm=TRUE) != 0, with=FALSE] ## drop const cols
  cols <- setdiff(names(DT),c(response))
  for (c in cols) {
    DT[,(c):=scale(DT[[c]])]
  }
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

validLoss <- function(x) {
  sqrt(mean((x-valid_y)^2))
}

score <- function(model, preds) {
  rmse = apply(preds, 2, validLoss)
  r = cbind(lambda=log10(rev(model$lambda)), rmse)
  plot(r)
  print(paste0("RMSE: ", min(rmse)))
  idx = which(rmse==min(rmse))
  print(paste0("lambda: ", (rev(model$lambda)[idx])))
}

## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, cutoff=FALSE,
                 params=list(rel_tol=1e-4, abs_tol=1e-4, rho=1e-5, max_iter=10000, adaptive_rho=FALSE, equil=TRUE))
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")

  print("POGS GPU: ")
  print(e1-s1)
  
  score(pogs, pogs_pred_y)
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
  
  score(glmnet, glmnet_pred_y)
}


## H2O
if (h2o) {
  h2o.init(nthreads=-1)
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

