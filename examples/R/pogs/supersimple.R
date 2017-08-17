library(h2ogpuml)

y <- c(3, 3.5, 6)
x <- as.matrix(data.frame(list(c(1, 3, 4),y)))

alpha <- 0.5

#model = h2ogpumlnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-10, nlambda=1000, cutoff=FALSE, params=list(max_iter=100000, abs_tol=1e-5, rel_tol=1e-5))
#model = h2ogpumlnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-6, intercept=TRUE, params=list(max_iter=2500,abs_tol=1e-5,rel_tol=1E-5))

#model = h2ogpumlnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-5, intercept=TRUE)
#print(paste0("RMSEH2OGPUML:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))

library(glmnet)
model = glmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL)
print(paste0("GLMNETSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))
model = glmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, standardize=FALSE)
print(paste0("GLMNETNOSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))


library(h2o)
h2o.init(nthreads=-1)
df.hex <- as.h2o(x)
model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE)
print(paste0("H2OSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, df.hex)[,1])-y)^2))))
model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE, standardize=FALSE)
print(paste0("H2ONOSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, df.hex)[,1])-y)^2))))
