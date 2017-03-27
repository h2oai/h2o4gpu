library(h2o)
library(pogs)
library(glmnet)
library(data.table)


response <- 'INCEARN'
family <- "gaussian"
#family <- "binomial"
pogs  <-TRUE
glmnet<-TRUE
h2o   <-TRUE
alpha <- 0.5 ## Lasso

file <- "/tmp/train.csv"

if (FALSE) {
  ## DATA PREP
  # created with 'head -n 4000001 ipums_2000-2015.csv > ipums_2000-2015_head4M.csv'
  f <- "~/ipums_2000-2015_head4M.csv"
  df <- fread(f)
  N <- nrow(df)
  N <- 0.5*N

  #df[['ID']] <- NULL ## ignore ID

  ## label encoding ## FIXME: leads to NAs?
  df[is.na(df)] <- 0
  #feature.names <- setdiff(names(df), response)
  #for (f in feature.names) {
  #  if (class(df[[f]])=="character") {
  #    levels <- unique(c(df[[f]]))
  #    df[[f]] <- as.integer(factor(df[[f]], levels=levels))
  #  }
  #}
  df <- df[1:N,sapply(df, is.numeric), with=FALSE]
  df <- df[,apply(df, 2, var, na.rm=TRUE) != 0, with=FALSE] ## drop const cols
  #df <- data.table(scale(as.data.frame(df)))
  fwrite(df, file)
  q()
} else {
  df <- fread(file)
  N <- nrow(df)
  summary(df)
}
H <- round(0.8*N)

train <- df[1:H,]
valid <- df[(H+1):N,]
cols <- setdiff(names(df), c(response))
train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
train_y  <- as.numeric(as.vector(train[[response]]))
valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
valid_y  <- as.numeric(as.vector(valid[[response]]))


## H2O used to compute the metrics
h2o.init(nthreads=-1)
df.hex <- h2o.importFile(file)
summary(df.hex)
train.hex <- df.hex[1:H,]
valid.hex <- df.hex[(H+1):N,]
if (family=="binomial") {
  train.hex[[response]] <- as.factor(train.hex[[response]])
  valid.hex[[response]] <- as.factor(valid.hex[[response]])
}


## POGS GPU
if (pogs) {
  s1 <- proc.time()
  pogs = pogsnet(x = train_x, y = train_y, family = family, alpha = alpha, lambda=c(0)) ## TODO: disable L1/L2 in the backend when passing lambda=0
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
  glmnet = glmnet(x = train_x, y = y, family = family, alpha = alpha, lambda=c(0))
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
  h2omodel <- h2o.glm(x=cols, y=response, training_frame=train.hex, family=family, alpha = alpha, lambda=0)
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
