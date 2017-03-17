library(h2o)
library(pogs)
library(glmnet)
library(data.table)

# kaggle springleaf
N<-10000
H <- 0.8*N
response <- 'target'
f <- "/tmp/train.csv"


## DATA PREP
df <- fread(f, nrows=N)
df <- df[,sapply(df, is.numeric), with=FALSE]
df[is.na(df)] <- 0

file <- paste0("/tmp/train.",N,".csv")
fwrite(df, file)
print(file)

h2o.init(nthreads=-1)
df.hex <- h2o.importFile(file)

train.hex <- df.hex[1:H,]
train.hex[[response]] <- as.factor(train.hex[[response]])
valid.hex <- df.hex[(H+1):N,]
valid.hex[[response]] <- as.factor(valid.hex[[response]])

train <- df[1:H,]
valid <- df[(H+1):N,]
cols <- setdiff(names(df), c(response))

train_x  <- as.matrix(as.data.frame(train[,cols,with=FALSE]))
train_y  <- as.vector(train[[response]])
valid_x  <- as.matrix(as.data.frame(valid[,cols,with=FALSE]))
valid_y  <- as.vector(valid[[response]])

## POGS GPU
s1 <- proc.time()
pogs = pogsnet(x = train_x, y = train_y, family = "binomial", alpha = 0.5, nlambda=2)
e1 <- proc.time()
pogs_pred_y = predict(pogs, valid_x, type="response")

print("POGS GPU: ")
print(e1-s1)
pogspreds <- as.h2o(pogs_pred_y)
h2o.auc(h2o.make_metrics(pogspreds[,1],   valid.hex[[response]]))



## GLMNET
s2 <- proc.time()
glmnet = glmnet(x = train_x, y = train_y, family = "binomial", alpha = 0.5, lambda=c(1e-5))
e2 <- proc.time()
glmnet_pred_y = predict(glmnet, valid_x, type="response")

print("GLMNET CPU")
print(e2-s2)
glmnetpreds <- as.h2o(glmnet_pred_y)
h2o.auc(h2o.make_metrics(glmnetpreds[,1], valid.hex[[response]]))



## H2O

s3 <- proc.time()
h2omodel <- h2o.glm(x=cols, y=response, training_frame=train.hex, family="binomial", lambda=c(1e-5))
e3 <- proc.time()
h2opreds <- h2o.predict(h2omodel, valid.hex)

print("H2O CPU ")
print(e3-s3)
h2o.auc(h2o.make_metrics(h2opreds[,3],    valid.hex[[response]]))
