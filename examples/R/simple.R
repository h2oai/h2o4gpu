library(pogs)
alpha <- 0

if (FALSE) {
age     <- c(4, 8, 7, 12, 6, 9, 10, 14, 7)
gender  <- as.factor(c(1, 0, 1, 1, 1, 0, 1, 0, 0))
bmi_p   <- c(0.86, 0.45, 0.99, 0.84, 0.85, 0.67, 0.91, 0.29, 0.88)
m_edu   <- as.factor(c(0, 1, 1, 2, 2, 3, 2, 0, 1))
p_edu   <- as.factor(c(0, 2, 2, 2, 2, 3, 2, 0, 0))
f_color <- as.factor(c("blue", "blue", "yellow", "red", "red", "yellow", "yellow", "red", "yellow"))
asthma <- c(1, 1, 0, 1, 0, 0, 0, 1, 1)
y <- asthma


xfactors <- model.matrix(asthma ~ gender + m_edu + p_edu + f_color)[, -1]
x        <- as.matrix(data.frame(age, bmi_p, xfactors))
}


df = read.csv("iris.csv")
x = as.matrix(df[,1:3])
y = df[,4]
print(x)
print(y)

#model = pogsnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-10, nlambda=1000, cutoff=FALSE, params=list(max_iter=100000, abs_tol=1e-5, rel_tol=1e-5))
#model = pogsnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, lambda.min.ratio=1e-7, intercept=TRUE, params=list(max_iter=2500))

intercept=TRUE

model = pogsnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, intercept=intercept, cutoff=FALSE, params=list(max_iter=100000, abs_tol=1e-5, rel_tol=1e-5))
print(model)
print(paste0("RMSEPOGS for s:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))

library(glmnet)
model = glmnet(x = x, y = y, family = "gaussian", alpha = alpha, lambda=NULL, standardize=FALSE, intercept=intercept)
print(model)
print(paste0("GLMNET:",sqrt(mean((predict(model, s=model$lambda[-1], newx=x)-y)^2))))
#
#library(h2o)
#h2o.init(nthreads=-1)
#df.hex <- as.h2o(cbind(x, y))
#model = h2o.glm(training_frame=df.hex, x = 1:ncol(x), y = ncol(df.hex), family = "gaussian", alpha = alpha, lambda_search=TRUE)
#print(paste0("H2O:",mean((as.data.frame(h2o.predict(model, df.hex)[,1])-y)^2)))
