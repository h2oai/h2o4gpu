library(h2oaiglm)

age     <- c(4, 8, 7, 12, 6, 9, 10, 14, 7)
gender  <- as.factor(c(1, 0, 1, 1, 1, 0, 1, 0, 0))
bmi_p   <- c(0.86, 0.45, 0.99, 0.84, 0.85, 0.67, 0.91, 0.29, 0.88)
m_edu   <- as.factor(c(0, 1, 1, 2, 2, 3, 2, 0, 1))
p_edu   <- as.factor(c(0, 2, 2, 2, 2, 3, 2, 0, 0))
f_color <- as.factor(c("blue", "blue", "yellow", "red", "red", "yellow", "yellow", "red", "yellow"))
asthma <- c(1, 1, 0, 1, 0, 0, 0, 1, 1)
#weights <- (1.0/9.0)*c(1, 1, 1, 1, 1, 1, 1, 1, 1)

alpha <- 0

xfactors <- model.matrix(asthma ~ gender + m_edu + p_edu + f_color)[, -1]
x        <- as.matrix(data.frame(age, bmi_p, xfactors))
y <- asthma

#model = h2oaiglmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-10, nlambda=1000, cutoff=FALSE, params=list(max_iter=100000, abs_tol=1e-5, rel_tol=1e-5))
#model = h2oaiglmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-6, intercept=TRUE, params=list(max_iter=2500,abs_tol=1e-5,rel_tol=1E-5))

model = h2oaiglmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-15, intercept=TRUE, noweight=FALSE, cutoff=FALSE, params=list(max_iter=3000,abs_tol=1e-15,rel_tol=1E-10,verbose=4,rho=0.1))
bestrmse=1E30
ii=0
bestii=-1
for (si in model$lambda){
    thislambda=si
    thisrmse=sqrt(mean((predict(model, s=si, newx=x)-y)^2))
    if(bestrmse>thisrmse){
        bestlambda=si
        bestrmse=thisrmse
        bestii=ii
    }
    ii=ii+1
    print(paste0("LAMBDA1look:",si))
    print(paste0("RMSE1look:",thisrmse))
}
print(paste0("RMSEH2OAIGLM:",bestrmse))
print(paste0("LAMBDAH2OAIGLM:",bestlambda))
print(paste0("IIH2OAIGLM:",bestii))

#model = h2oaiglmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-10, intercept=FALSE, noweight=TRUE)
#for (si in model$lambda){
#    print(paste0("RMSENOINTH2OAIGLM:",sqrt(mean((predict(model, s=si, newx=x)-y)^2))))
#}

library(glmnet)
#model = glmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL)
#print(paste0("GLMNETSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))
#model = glmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, nlambda=1000, lambda.min.ratio = 1e-10, standardize=FALSE)
model = glmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, nlambda=1000, lambda=c(246,24,2,0.1,0.01,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-9), standardize=FALSE)
#print(paste0("GLMNETNOSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))
bestrmse=1E30
ii=0
bestii=-1
for (si in model$lambda){
    thislambda=si
    thisrmse=sqrt(mean((predict(model, s=si, newx=x)-y)^2))
    if(bestrmse>thisrmse){
        bestlambda=si
        bestrmse=thisrmse
        bestii=ii
    }
    ii=ii+1
    print(paste0("LAMBDA2look:",si))
    print(paste0("RMSE2look:",thisrmse))
}
print(paste0("RMSEGLMNET:",bestrmse))
print(paste0("LAMBDAGLMNET:",bestlambda))
print(paste0("IIGLMNET:",bestii))



library(h2o)
h2o.init(nthreads=-1)
df.hex <- as.h2o(cbind(x, asthma))
#model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE)
#print(paste0("H2OSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, s=model$lambda[-1], df.hex)[,1])-y)^2))))
model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE, standardize=FALSE)
print(paste0("H2ONOSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, s=model$lambda[-1], df.hex)[,1])-y)^2))))
#3print(paste0("LAMBDAH2O:",model$lambda[-1]))

