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
alpha <- 1 ## Lasso
lambda <- c(0) ## no regularization

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

h2o.init(nthreads=-1)
df.hex <- h2o.importFile(file)
summary(df.hex)

train.hex <- df.hex[1:H,]
valid.hex <- df.hex[(H+1):N,]
if (family=="binomial") {
  train.hex[[response]] <- as.factor(train.hex[[response]])
  valid.hex[[response]] <- as.factor(valid.hex[[response]])
}

train <- DT[1:H,]
valid <- DT[(H+1):N,]

train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
train_y  <- as.numeric(as.vector(train[[response]]))
valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
valid_y  <- as.numeric(as.vector(valid[[response]]))


## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=NULL, params=list(max_iter=1000))
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")

  print("POGS GPU: ")
  print(e1-s1)
  pogspreds <- as.h2o(pogs_pred_y)
  summary(pogspreds)
  if (family == "gaussian") {
    print(h2o.rmse(h2o.make_metrics(pogspreds[,1], valid.hex[[response]])))
  } else {
    print(h2o.auc(h2o.make_metrics(pogspreds[,1], valid.hex[[response]])))
  }
}


## GLMNET
if (glmnet) {
  if (family=="binomial") {
    y = as.factor(train_y)
  } else {
    y = train_y
  }
  s2 <- proc.time()
  glmnet = glmnet(x = train_x, y = y, family = family, alpha = alpha, lambda=lambda)
  e2 <- proc.time()
  glmnet_pred_y = predict(glmnet, valid_x, type="response")

  print("GLMNET CPU")
  print(e2-s2)
  glmnetpreds <- as.h2o(glmnet_pred_y)
  summary(glmnetpreds)
  if (family == "gaussian") {
    print(h2o.rmse(h2o.make_metrics(glmnetpreds[,1], valid.hex[[response]])))
  } else {
    print(h2o.auc(h2o.make_metrics(glmnetpreds[,1], valid.hex[[response]])))
  }
}


## H2O
if (h2o) {
  s3 <- proc.time()
  h2omodel <- h2o.glm(x=cols, y=response, training_frame=train.hex, family=family, alpha = alpha, lambda_search=FALSE, lambda=lambda)
  e3 <- proc.time()
  h2opreds <- h2o.predict(h2omodel, valid.hex)
  summary(h2opreds)

  print("H2O CPU ")
  print(e3-s3)
  if (family == "gaussian") {
    print(h2o.rmse(h2o.make_metrics(h2opreds[,1], valid.hex[[response]])))
  } else {
    print(h2o.auc(h2o.make_metrics(h2opreds[,3], valid.hex[[response]])))
  }
}

#[1] "POGS GPU: "
#   user  system elapsed 
#  2.916   1.288   4.204 
#  |======================================================================| 100%
#[1] 0.4814968
#
#[1] "GLMNET CPU"
#   user  system elapsed 
#431.420   0.684 432.133 
#  |======================================================================| 100%
#[1] 0.3883085
#
#[1] "H2O CPU "
#   user  system elapsed 
#  2.236   0.116  55.697 
#[1] 0.3890118
