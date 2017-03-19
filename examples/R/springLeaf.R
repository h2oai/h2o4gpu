library(h2o)
library(pogs)
library(glmnet)
library(data.table)

#https://www.kaggle.com/c/springleaf-marketing-response/data
N<-145231 ## max
#N<-1000  ## ok for accuracy tests
H <- round(0.8*N) ## need to split into train/test since kaggle test set has no labels
#f <- "gunzip -c ../data/springleaf/train.csv.zip"
f <- "~/kaggle/springleaf/input/train.csv"

response <- 'target'
family <- "gaussian"
#family <- "binomial"
pogs  <-TRUE
glmnet<-TRUE
h2o   <-TRUE


## DATA PREP
df <- fread(f, nrows=N)

df[['ID']] <- NULL ## ignore ID

## label encoding ## FIXME: leads to NAs?
df[is.na(df)] <- 0
#feature.names <- setdiff(names(df), response)
#for (f in feature.names) {
#  if (class(df[[f]])=="character") {
#    levels <- unique(c(df[[f]]))
#    df[[f]] <- as.integer(factor(df[[f]], levels=levels))
#  }
#}
df <- df[,sapply(df, is.numeric), with=FALSE]
#df <- data.table(scale(as.data.frame(df)))

file <- paste0("/tmp/train.",N,".csv")
fwrite(df, file)

h2o.init(nthreads=-1)
df.hex <- h2o.importFile(file)
summary(df.hex)

train.hex <- df.hex[1:H,]
valid.hex <- df.hex[(H+1):N,]
if (family=="binomial") {
  train.hex[[response]] <- as.factor(train.hex[[response]])
  valid.hex[[response]] <- as.factor(valid.hex[[response]])
}

train <- df[1:H,]
valid <- df[(H+1):N,]
cols <- setdiff(names(df), c(response))

train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
train_y  <- as.numeric(as.vector(train[[response]]))
valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
valid_y  <- as.numeric(as.vector(valid[[response]]))


## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = cv.pogsnet(x = train_x, y = train_y, family = family, alpha = 0.5)
  e1 <- proc.time()
  pogs_pred_y = predict(pogs$pogsnet.fit, valid_x, type="response")

  print("POGS GPU: ")
  print(e1-s1)
  pogspreds <- as.h2o(pogs_pred_y)
  if (family == "gaussian") {
    h2o.rmse(h2o.make_metrics(pogspreds[,1], valid.hex[[response]]))
  } else {
    h2o.auc(h2o.make_metrics(pogspreds[,1], valid.hex[[response]]))
  }
}


## GLMNET
if (glmnet) {
  require(doMC)
  registerDoMC(cores=10)
  if (family=="binomial") {
    y = as.factor(train_y)
  } else {
    y = train_y
  }
  s2 <- proc.time()
  glmnet = cv.glmnet(nfolds=10, parallel=TRUE, x = train_x, y = y, family = family, alpha = 0.5)
  e2 <- proc.time()
  glmnet_pred_y = predict(glmnet$glmnet.fit, valid_x, type="response")

  print("GLMNET CPU")
  print(e2-s2)
  glmnetpreds <- as.h2o(glmnet_pred_y)
  if (family == "gaussian") {
    h2o.rmse(h2o.make_metrics(glmnetpreds[,1], valid.hex[[response]]))
  } else {
    h2o.auc(h2o.make_metrics(glmnetpreds[,1], valid.hex[[response]]))
  }
}


## H2O
if (h2o) {
  s3 <- proc.time()
  h2omodel <- h2o.glm(nfolds=10, x=cols, y=response, training_frame=train.hex, family=family, alpha = 0.5, lambda_search=TRUE)
  e3 <- proc.time()
  h2opreds <- h2o.predict(h2omodel, valid.hex)

  print("H2O CPU ")
  print(e3-s3)
  if (family == "gaussian") {
    h2o.rmse(h2o.make_metrics(h2opreds[,1], valid.hex[[response]]))
  } else {
    h2o.auc(h2o.make_metrics(h2opreds[,3], valid.hex[[response]]))
  }
}


### Results i7-5820k / Titan-X Pascal
### Elastic Net full regularization path with 10-fold CV

#POGS GPU
#   user  system elapsed 
#390.832  61.132 452.642 
#rmse 0.4220073

#GLMNET CPU - multithreaded
#     user    system   elapsed 
#14039.984    10.092  2147.720 
#rmse 0.4220078

#H2O CPU (IRLSM)
#     user    system   elapsed 
#  270.644    22.264 22098.499
#rmse 0.3879998


