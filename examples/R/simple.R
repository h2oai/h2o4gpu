library(h2oaiglm)

age     <- c(4, 8, 7, 12, 6, 9, 10, 14, 7)
gender  <- as.factor(c(1, 0, 1, 1, 1, 0, 1, 0, 0))
bmi_p   <- c(0.86, 0.45, 0.99, 0.84, 0.85, 0.67, 0.91, 0.29, 0.88)
m_edu   <- as.factor(c(0, 1, 1, 2, 2, 3, 2, 0, 1))
p_edu   <- as.factor(c(0, 2, 2, 2, 2, 3, 2, 0, 0))
f_color <- as.factor(c("blue", "blue", "yellow", "red", "red", "yellow", "yellow", "red", "yellow"))
asthma <- c(1, 1, 0, 1, 0, 0, 0, 1, 1)

alpha <- 0.5

xfactors <- model.matrix(asthma ~ gender + m_edu + p_edu + f_color)[, -1]
x        <- as.matrix(data.frame(age, bmi_p, xfactors))
y <- asthma

#model = h2oaiglmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-10, nlambda=1000, cutoff=FALSE, params=list(max_iter=100000, abs_tol=1e-5, rel_tol=1e-5))
#model = h2oaiglmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-6, intercept=TRUE, params=list(max_iter=2500,abs_tol=1e-5,rel_tol=1E-5))

model = h2oaiglmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-5, intercept=TRUE)
print(paste0("RMSEH2OAIGLM:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))

library(glmnet)
model = glmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL)
print(paste0("GLMNETSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))
model = glmnet(x = x, y = asthma, family = "gaussian", alpha = alpha, lambda=NULL, standardize=FALSE)
print(paste0("GLMNETNOSTDRMSE:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))


library(h2o)
h2o.init(nthreads=-1)
df.hex <- as.h2o(cbind(x, asthma))
model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE)
print(paste0("H2OSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, s=model$lambda[-1], df.hex)[,1])-y)^2))))
model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE, standardize=FALSE)
print(paste0("H2ONOSTDRMSE:",sqrt(mean((as.data.frame(h2o.predict(model, s=model$lambda[-1], df.hex)[,1])-y)^2))))
