library(h2o)
library(h2ogpuml)
library(glmnet)
library(data.table)

#https://www.kaggle.com/c/springleaf-marketing-response/data
N<-145231 ## max
N<-1000  ## ok for accuracy tests
H <- round(0.8*N) ## need to split into train/test since kaggle test set has no labels
#f <- "gunzip -c ../data/springleaf/train.csv.zip"
f <- "~/kaggle/springleaf/input/train.csv"

response <- 'target'
family <- "gaussian"
#family <- "binomial"
h2ogpuml  <-TRUE
glmnet<-TRUE
h2o   <-TRUE
alpha <- 1 ## Lasso
nfolds <- 10

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


## H2OGPUML GPU
if (h2ogpuml) {
  s1 <- proc.time()
  h2ogpuml = cv.h2ogpumlnet(nfolds=nfolds, x = train_x, y = train_y, family = family, alpha = alpha, cutoff=FALSE)
  print(paste0("lambda_1se=",h2ogpuml$lambda.1se))
  h2ogpuml_pred_y = predict(h2ogpuml$h2ogpumlnet.fit, s=h2ogpuml$lambda.1se, valid_x, type="response")
  e1 <- proc.time()

  print("H2OGPUML GPU: ")
  print(e1-s1)
  h2ogpumlpreds <- as.h2o(h2ogpuml_pred_y)
  summary(h2ogpumlpreds)
  if (family == "gaussian") {
    print(h2o.rmse(h2o.make_metrics(h2ogpumlpreds[,1], valid.hex[[response]])))
  } else {
    print(h2o.auc(h2o.make_metrics(h2ogpumlpreds[,1], valid.hex[[response]])))
  }
}


## GLMNET
if (glmnet) {
  require(doMC)
  registerDoMC(cores=nfolds)
  if (family=="binomial") {
    y = as.factor(train_y)
  } else {
    y = train_y
  }
  s2 <- proc.time()
  glmnet = cv.glmnet(nfolds=nfolds, parallel=TRUE, x = train_x, y = y, family = family, alpha = alpha)
  e2 <- proc.time()
  print(paste0("lambda_1se=",glmnet$lambda.1se))
  glmnet_pred_y = predict(glmnet$glmnet.fit, s=glmnet$lambda.1se, valid_x, type="response")

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
  h2omodel <- h2o.glm(nfolds=nfolds, x=cols, y=response, training_frame=train.hex, family=family, alpha = alpha, lambda_search=TRUE, solver="COORDINATE_DESCENT_NAIVE")
  regpath = h2o.getGLMFullRegularizationPath(h2omodel)
  n = dim(regpath$coefficients)[1]
  coefs = NULL
  for (i in 1:n) {
    if (h2omodel@model$lambda_1se == regpath$lambdas[i]) {
      print(paste0("lambda_1se=",regpath$lambdas[i]))
      coefs = regpath$coefficients[i,]
      break
    }
  }
  h2omodel2 <- h2o.makeGLMModel(h2omodel,coefs)
  e3 <- proc.time()
  h2opreds <- h2o.predict(h2omodel2, valid.hex)
  summary(h2opreds)

  print("H2O CPU ")
  print(e3-s3)
  if (family == "gaussian") {
    print(h2o.rmse(h2o.make_metrics(h2opreds[,1], valid.hex[[response]])))
  } else {
    print(h2o.auc(h2o.make_metrics(h2opreds[,3], valid.hex[[response]])))
  }
}


### 569674a1dfa i7-5820k / Titan-X Pascal
### Elastic Net full regularization path with 10-fold CV

#H2OGPUML GPU
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




### e3b7d0f6f1c0 Dual Xeon / GTX1080
### Elastic Net full regularization path with 10-fold CV

#H2OGPUML GPU
#lambda_1se=2105060.83223774
#   user  system elapsed
#532.176 114.980 647.744
#rmse 0.4118603

#GLMNET CPU
#lambda_1se=0.000515305983455443
#    user   system  elapsed
#7144.016   81.388 1521.256
#rmse 0.3886783

#H2O CPU
#    user   system  elapsed
#lambda_1se=0.000900507965438238
#     user    system   elapsed
#  359.588    61.212 14413.293
#rmse 0.3890318



### Latest timing for f8d241e on mr-dl1

#H2OGPUML GPU
#lambda_1se=27340705.2636287
#   user  system elapsed 
#216.668  49.136 265.898 
#rmse 0.422007

#H2OGPUML CPU
#lambda_1se=7432811.98777276
#     user    system   elapsed 
#25356.744  8514.276   993.101 
#rmse 0.4166237

#GLMNET CPU
#lambda_1se=0.000542334850133602
#    user   system  elapsed
#7739.856   14.752 1823.866
#rmse 0.3892373

## H2O with COORDINATE_DESCENT_NAIVE (tomas_COD_update branch 8d0220a02724a6c)
#lambda_1se=0.00104014509806017
#   user  system elapsed 
#  9.392   1.024 727.941 
#rmse 0.3901852



## Latest timing for 5cb02b84a on ovaclokka
## Titan-X Pascal / i7 5820k

#H2OGPUML GPU
#   user  system elapsed 
#167.172  26.736 193.956 
#rmse 0.422007

#H2OGPUML CPU
#    user   system  elapsed 
#8813.880  622.592  875.563 
#rmse 0.4166237

#GLMNET CPU
#     user    system   elapsed 
#16185.664    11.280  2541.528 
#rmse 0.3888805

#H2O CPU (COD)
#    user   system  elapsed 
#   9.428    0.380 1154.257 
#rmse 0.3901853
