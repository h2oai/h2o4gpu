library(h2o)
library(pogs)
library(glmnet)
library(data.table)

#https://www.kaggle.com/c/springleaf-marketing-response/data
N<-145231 ## max
N<-10000  ## ok for accuracy tests
H <- round(0.8*N) ## need to split into train/test since kaggle test set has no labels
#f <- "gunzip -c ../data/springleaf/train.csv.zip"
f <- "gunzip -c ~/kaggle/springleaf/input/train.csv.zip"

response <- 'target'
family <- "gaussian"
#family <- "binomial" ## FIXME: Buggy?
pogs<-TRUE
glmnet<-TRUE
h2o<-TRUE


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
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = 0.5, lambda=c(0)) #TODO: Disable lambda search (no longer use nlambda=100)
  e1 <- proc.time()
  pogs_pred_y = predict(pogs, valid_x, type="response")
  pogs_pred_y = pogs_pred_y / norm(pogs_pred_y)

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
  s2 <- proc.time()
  glmnet = glmnet(x = train_x, y = train_y, family = family, alpha = 0.5, lambda=c(0))
  e2 <- proc.time()
  glmnet_pred_y = predict(glmnet, valid_x, type="response")

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
  h2omodel <- h2o.glm(x=cols, y=response, training_frame=train.hex, family=family, alpha = 0.5, solver="IRLSM", lambda=c(0))
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

# Results on dual Intel(R) Xeon(R) CPU E5-2687W and Titan X (Pascal)
# dual Xeon costs about $1300 and 1 Titan costs about $1300.
# N=140000 train/test: 80%/20%
# pogs-gpu: 50s
# h2o-cpu: 111s
# Perf/Price boost with gpu: 2.22X
