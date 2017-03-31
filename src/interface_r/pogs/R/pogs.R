############################################################################################ 
###################################### POGS INTERFACE ######################################
############################################################################################
#' @title Proximal Operator Graph Solver
#' @description Solver for convex optimization problems in the form
#' \deqn{\min. f(y) + g(x), \textrm{ s.t. } y = Ax}{min. f(y) + g(x),  s.t. y = Ax,}
#' where \eqn{f} and \eqn{g} are convex, separable, and take the form
#' \deqn{c h(a x - b) + d x + e x^2,} where \eqn{a}, \eqn{b} and
#' \eqn{d} are real, \eqn{c} and \eqn{d} are non-negative and \eqn{h} is one
#' of 16 convex functions (see. \link{kAbs}, \link{kExp}, \link{kHuber},
#' \link{kIdentity}, \link{kIndBox01}, \link{kIndEq0}, \link{kIndGe0},
#' \link{kIndLe0}, \link{kLogistic}, \link{kMaxNeg0}, \link{kMaxPos0},
#' \link{kNegEntr}, \link{kNegLog}, \link{kRecipr}, \link{kSquare}, \link{kZero}).
#' @param A Matrix encoding constraint \eqn{y = Ax}.
#' @param f List with fields a, b, c, d, e, and h. All fields except h are
#' optional and each field which is specified must be a vector of length 1 or nrow(A).
#' @param g List with fields a, b, c, d, e, and h. All fields except h are
#' optional and each field which is specified must be a vector of length 1 or ncol(A).
#' @param params List of parameters (rel_tol=1e-3, abs_tol=1e-4, rho=1.0,
#' max_iter=1000, quiet=FALSE, adaptive_rho=TRUE, equil=TRUE).
#' All parameters are optional and take on a default value if not specified.
#' @examples
#' # Specify Lasso problem.
#' A = matrix(rnorm(100 * 10), 100, 10)
#' b = rnorm(100)
#' lambda = 5
#' f = list(h = kSquare(), b = b)
#' g = list(h = kAbs(), c = lambda)
#' pogs(A, f, g)
#' @useDynLib pogs
#' @import Matrix
#' @export  
pogs <- function(A, f, g, params=list()) {
  # Make sure A is numeric matrix.
  if (!is.numeric(A)) {
    stop("pogs(): input A must be numeric!")
  }
  if (!is.matrix(A)) {
    stop("pogs(): input A must be a matrix!")
  }

  # Make sure f and g are lists.
  if (!is.list(f) || !is.list(g) || !is.list(params)) {
    stop("pogs(): inputs f, g and params must be lists!")
  }

  # Make f and g into lists if they are not already
  if (length(f) == 0 || length(g) == 0) {
    stop("Neither f nor g may not be empty")
  }
  if (!is.list(f[[1]]) && !is.list(g[[1]])) {
    f = list(f)
    g = list(g)
  } else if (!is.list(f[[1]]) || !is.list(g[[1]])) {
    stop("Either both or neither of f and g must be nested lists")
  } else if (is.list(f[[1]]) && is.list(g[[1]]) && length(f) != length(g)) {
    stop("Length of f and g must be the same")
  }

  # Check fields in f.
  for (i in 1:length(f)) {
    fi = f[[i]]
    if (is.null(names(fi))) {
      stop("f must be a named list (elements h, a, b, c, d, e)")
    }
    if (!any(names(fi) == "h")) {
      stop("pogs(): field f$h is required!")
    }
    for (name in names(fi)) {
      if (!any(name == c("a", "b", "c", "d", "e", "h"))) {
        stop(cat("pogs(): field f$", name, " unknown!", sep=""))
      }
      if (!is.numeric(fi[[name]])) {
        stop(cat("pogs(): field f$", name, " must be numeric!", sep=""))
      }
      # Convert to double (in case it's a matrix).
      fi[[name]] = as.double(fi[[name]])
      if (!is.vector(fi[[name]])) {
        stop(cat("pogs(): field f$", name, " must be a vector!", sep=""))
      }
      if (length(fi[[name]]) != nrow(A) && length(fi[[name]]) != 1) {
        stop(cat("pogs(): field f$", name, " must must have length nrow(A) or 1!", sep=""))
      }
    }
  }

  # Check fields in g.
  for (i in 1:length(g)) {
    gi = g[[i]]
    if (is.null(names(gi))) {
      stop("g must be a named list (elements h, a, b, c, d, e)")
    }
    if (!any(names(gi) == "h")) {
      stop("pogs(): field g$h is required!")
    }
    for (name in names(gi)) {
      if (!any(name == c("a", "b", "c", "d", "e", "h"))) {
        stop(cat("pogs(): field g$", name, " unknown!", sep=""))
      }
      if (!is.numeric(gi[[name]])) {
        stop(cat("pogs(): field g$", name, " must be numeric!", sep=""))
      }
      # Convert to double (in case it's a matrix).
      gi[[name]] = as.double(gi[[name]])
      if (!is.vector(gi[[name]])) {
        stop(cat("pogs(): field g$", name, " must be a vector!", sep=""))
      }
      if (length(gi[[name]]) != ncol(A) && length(gi[[name]]) != 1) {
        stop(cat("pogs(): field g$", name, " must must have length ncol(A) or 1!", sep=""))
      }
    }
  }
 
  # Check fields in params.
  if (length(params) > 0 && is.null(names(params))) {
    stop("params must be a named list (elements abs_tol, rel_tol, rho, max_iter, verbose, adaptive_rho, equil, gap_stop)")
  }
  for (name in names(params)) {
    if (!any(name == c("rel_tol", "abs_tol", "rho", "max_iter", "verbose", "adaptive_rho", "equil", "gap_stop"))) {
      stop(cat("pogs(): field params$", name, " unknown!", sep=""))
    }
    if (!is.numeric(params[[name]]) && name != "adaptive_rho" && name != "equil" && name != "gap_stop") {
      stop(cat("pogs(): field params$", name, " must be numeric!", sep=""))
    }
    if (!is.logical(params[[name]]) && (name == "adaptive_rho" || name == "equil" || name == "gap_stop")) {
      stop(cat("pogs(): field params$", name, " must be logical!", sep=""))
    }
    if (length(params[[name]]) != 1) {
      stop(cat("pogs(): field params$", name, " must must have length 1!", sep=""))
    }
    if (name == "verbose") {
      params[[name]] = as.integer(params[[name]])
    }
  }

  # Call POGS.
  .Call("PogsWrapper", A, f, g, params)
}

############################################################################################ 
################################# POGS OBJECTIVE FUNCTIONS #################################
############################################################################################
#' @title Absolute Value Function
#' @description \deqn{h(x) = |x|}
#' @param m Number of times that the function should be repeated.
#' @export
kAbs <- function(m=1) {
  rep(0, m)
}

#' @title Exponential Function
#' @description \deqn{h(x) = e^x}
#' @param m Number of times that the function should be repeated.
#' @export
kExp <- function(m=1) {
  rep(1, m)
}

#' @title Huber Function
#' @description \deqn{h(x) = huber(x)}
#' @param m Number of times that the function should be repeated.
#' @export
kHuber <- function(m=1) {
  rep(2, m)
}

#' @title Identity Function
#' @description \deqn{h(x) = x}
#' @param m Number of times that the function should be repeated.
#' @export
kIdentity <- function(m=1) {
  rep(3, m)
}

#' @title Indicator Function of [0, 1] Box
#' @description \deqn{h(x) = I(0 <= x <= 1)}
#' @param m Number of times that the function should be repeated.
#' @export
kIndBox01 <- function(m=1) {
  rep(4, m)
}

#' @title Indicator Function of Origin
#' @description \deqn{h(x) = I(x = 0)}
#' @param m Number of times that the function should be repeated.
#' @export
kIndEq0 <- function(m=1) {
  rep(5, m)
}

#' @title Indicator Function of Non-Negative Orthant
#' @description \deqn{h(x) = I(x >= 0)}
#' @param m Number of times that the function should be repeated.
#' @export
kIndGe0 <- function(m=1) {
  rep(6, m)
}

#' @title Indicator Function of Non-Positive Orthant
#' @description \deqn{h(x) = I(x <= 0)}
#' @param m Number of times that the function should be repeated.
#' @export
kIndLe0 <- function(m=1) {
  rep(7, m)
}

#' @title Logistic Function
#' @description \deqn{h(x) = log(1 + e^x)}
#' @param m Number of times that the function should be repeated.
#' @export
kLogistic <- function(m=1) {
  rep(8, m)
}

#' @title Positive-Part Function
#' @description \deqn{h(x) = max(0, x)}
#' @param m Number of times that the function should be repeated.
#' @export
kMaxPos0 <- function(m=1) {
  rep(9, m)
}

#' @title Negative-Part Function
#' @description \deqn{h(x) = max(0, -x)}
#' @param m Number of times that the function should be repeated.
#' @export
kMaxNeg0 <- function(m=1) {
  rep(10, m)
}

#' @title Negative Entropy Function
#' @description \deqn{h(x) = x log(x)}
#' @param m Number of times that the function should be repeated.
#' @export
kNegEntr <- function(m=1) {
  rep(11, m)
}

#' @title Negative Log Function
#' @description \deqn{h(x) = -log(x)}
#' @param m Number of times that the function should be repeated.
#' @export
kNegLog <- function(m=1) {
  rep(12, m)
}

#' @title Reciprocal Function
#' @description \deqn{h(x) = 1 / x}
#' @param m Number of times that the function should be repeated.
#' @export
kRecipr <- function(m=1) {
  rep(13, m)
}

#' @title Square Function
#' @description \deqn{h(x) = (1/2) x^2}
#' @param m Number of times that the function should be repeated.
#' @export
kSquare <- function(m=1) {
  rep(14, m)
}

#' @title Zero Function
#' @description \deqn{h(x) = 0}
#' @param m Number of times that the function should be repeated.
#' @export
kZero <- function(m=1) {
  rep(15, m)
}

############################################################################################ 
##################################### POGSNET FUNCTION #####################################
############################################################################################
#' @title Fit a GLM with lasso or elasticnet regularization using POGS
#' @description Fit a generalized linear model via penalized maximum likelihood.
#' See glmnet package for more detail.
#' @param x Input matrix.
#' @param y Response variable.
#' @param family Response type.
#' @param weights Observation weights.
#' @param alpha The elasticnet mixing parameter with \eqn{0 \le \alpha \le 1}. The penalty
#' is defined as \deqn{(1-\alpha)/2||\beta||_2^2 + \alpha ||\beta||_1.}
#' @param nlambda The number of \code{lambda} values.
#' @param lambda.min.ratio Smallest value for \code{lambda} as fraction of \code{lambda.max}.
#' @param lambda A user supplied \eqn{lambda} sequence.
#' @param penalty.factor Separate penalty factors can be applied to each coefficient.
#' @param intercept Should intercept be fitted.
#' @param params Pass list of parameters to solver
#' @param cutoff Discard values of lambda for which beta remains unchanged.
#' @export
pogsnet <- function(x, y, family=c("gaussian", "binomial"),
                    weights, alpha=1, nlambda=100,
                    lambda.min.ratio=ifelse(nobs < nvars, 0.01, 0.0001), lambda=NULL,
                    penalty.factor=rep(1, nvars), intercept=TRUE, params=list(),
                    cutoff=TRUE) {

  # Check Family
  family = match.arg(family)
  
  # Check Matrix x
  y = drop(y)
  nobs = dim(x)[1]
  nvars = dim(x)[2]
  if (nobs != length(y)) {
    stop(paste("Rows in x (", nobs, ") and y (", length(y), ") must be the same", sep=""))
  }
  
  # Check Intercept
  if (intercept) {
    x = cbind(rep(1, nobs), x)
    penalty.factor = c(0, penalty.factor)
  }

  # Check Alpha
  if(alpha > 1){
    warning("alpha > 1; set to 1")
    alpha = 1
  } else if(alpha < 0){
    warning("alpha < 0; set to 0")
    alpha = 0
  }
  alpha = as.double(alpha)
  
  # Check Weights
  if(missing(weights)) {
    weights = rep(1 / nobs, nobs)
  } else if(length(weights) != nobs) {
    stop(paste("Number of elements in weights (", length(weights), ") not equal to the number of rows of x (", nobs,")", sep=""))
  } else {
    weights = weights / sum(weights)
  }
  
  # Check Lambdas
  if(is.null(lambda) && lambda.min.ratio >= 1) {
    stop("lambda.min.ratio should be less than 1")
  } else if(any(lambda < 0)) {
    stop("lambdas should be non-negative")
  } else if (!is.null(lambda)) {
    nlambda = as.integer(length(lambda))
    lambda = rev(sort(lambda))
  }
  
  # Setup for solver
  f = list()
  g = list()
  if (family == "gaussian") {
    if (nlambda==1 && lambda[[1]] == 0) {
      f[[1]] = list(h = kSquare(), c = weights, b = y)
      g[[1]] = list(h = kZero())
      print("OLS")
    }
    else {
      if (is.null(lambda)) {
        if (intercept) {
          lambda.max = max(abs(t(x[,-1]) %*% (weights * (y - mean(y))))) / (alpha + 1e-3)
        } else {
          lambda.max = max(abs(t(x) %*% y)) / (alpha + 1e-3)
        }
        lambda.min = lambda.max * lambda.min.ratio
          lambda = rev(exp(seq(log(lambda.min), log(lambda.max), length=nlambda)))
      }
      for (i in 1:nlambda) {
        f[[i]] = list(h = kSquare(), c = weights, b = y)
          g[[i]] = list(h = kAbs(), c = alpha * lambda[i] * penalty.factor, e = (1 - alpha) * lambda[i] * penalty.factor)
      }
    }
  } else if (family == "binomial") {
    if (is.null(lambda)) {
      if (intercept) {
        x0 = log(sum(y * weights) / (1 - sum(y * weights)))
        lambda.max = max(abs(t(x[,-1]) %*% (weights * (exp(x0) / (1 + exp(x0)) - y)))) / (alpha + 1e-3)
      } else {
        lambda.max = max(abs(t(x) %*% (weights * (1 / 2 - y)))) / (alpha + 1e-3)
      }
      lambda.min = lambda.max * lambda.min.ratio
      lambda = rev(exp(seq(log(lambda.min), log(lambda.max), length=nlambda)))
    }
    for (i in 1:nlambda) {
      f[[i]] = list(h = kLogistic(), c = weights, d = -y * weights)
      g[[i]] = list(h = kAbs(), c = alpha * lambda[i] * penalty.factor, e = (1 - alpha) * lambda[i] * penalty.factor)
    }
  }
  
  # Solve
  soln = pogs(x, f, g, params) 
  
  # Extract Output
  fit = list(beta = matrix(rep(0, (nvars + 1) * nlambda), nvars + 1, nlambda),
             df = rep(0, nlambda),
             lambda = rep(0, nlambda),
             dev.ratio = rep(0, nlambda))
  last = nlambda
  if (cutoff && nlambda > 1) {
    for (i in 2:nlambda) {
      if (max(abs(soln$x[, i] - soln$x[, i - 1])) < 1e-4 * sum(abs(soln$x[, i]))) {
        last = i
        break
      }
    }
  }
  fit$beta = Matrix(soln$x[,1:last], sparse=TRUE)
  if (nlambda>1) {
    fit$df = apply(soln$x[-1,1:last], 2, function(x) { sum(x != 0) })
  } else {
    fit$df = NULL #sum(soln$x[-1,1:last]!=0) ## not sure what to do here
  }
  fit$lambda = lambda[1:last]
  fit$call = match.call()
  fit$dev.ratio = fit$dev.ratio[1:last]
  class(fit) = c("pogsnet", family)
  fit
}

# Copied from glmnet
lambda.interp <- function(lambda, s){  
  if(length(lambda) == 1) {
    nums = length(s)
    left = rep(1, nums)
    right = left
    sfrac = rep(1, nums)
  } else {
    s[s > max(lambda)] = max(lambda)
    s[s < min(lambda)] = min(lambda)
    k=length(lambda)
    sfrac = (lambda[1] - s) / (lambda[1] - lambda[k])
    lambda = (lambda[1] - lambda) / (lambda[1] - lambda[k])
    coord = approx(lambda, seq(lambda), sfrac)$y
    left = floor(coord)
    right = ceiling(coord)
    sfrac = (sfrac - lambda[right]) / (lambda[left] - lambda[right])
    sfrac[left == right] = 1
  }
  list(left=left, right=right, frac=sfrac)
}

#' @title Pogsnet prediction
#' @description Predict response for new value input
#' @param object Value returned by pogsnet(..).
#' @param newx New value of x for which prediction should be done.
#' @param s Values of lambda for which coefficient should be returned.
#' @param type Predict response or class.
#' @param ... Not used. Other arguments to predict.
#' @export
predict.pogsnet <- function(object, newx, s=NULL, type=c("link", "response", "class"), ...) {
  type = match.arg(type)
  if (is.null(s)) {
    beta = object$beta
  } else {
    lamlist = lambda.interp(object$lambda, s)
    beta = object$beta[, lamlist$left, drop=FALSE] %*% Diagonal(x=lamlist$frac) + 
      object$beta[, lamlist$right, drop=FALSE] %*% Diagonal(x=1 - lamlist$frac)
  }
  fitted = as.matrix(cbind2(1, newx) %*% beta)
  if (any(class(object) == "binomial")) {
    if (type == "response") {
      fitted = 1 / (1 + exp(-fitted))
    } else if (type == "class") {
      fitted = 1 / (1 + exp(-fitted)) > 1/2
    }
  }
  fitted
}

#' @title pogsnet coefficients.
#' @description Returns coeficients of pogsnet fit.
#' @param object Value returned by pogsnet(..).
#' @param s Values of lambda for which coefficient should be returned.
#' @param ... Not used. Other arguments to predict.
#' @export
coef.pogsnet <- function(object, s=NULL, ...) {
  if (is.null(s)) {
    coefs = object$beta
  } else {
    lamlist = lambda.interp(object$lambda, s)
    coefs = object$beta[, lamlist$left, drop=FALSE] %*% Diagonal(x=lamlist$frac) + 
      object$beta[, lamlist$right, drop=FALSE] %*% Diagonal(x=1 - lamlist$frac)
  }
  rownames(coefs) = c("(Intercept)", paste("V", seq(1, nrow(coefs) - 1), sep=""))
  coefs
}

#' @title pogsnet coefficient plot.
#' @description Plots coefficients
#' @param x pogsnet object.
#' @param xvar x-variable to plot coefficient against.
#' @param ... Extra parameters for matplot.
#' @export
plot.pogsnet <- function(x, xvar=c("norm","lambda","dev"), ...) {
  xvar = match.arg(xvar)
  if (xvar == "norm") {
    index = apply(abs(x$beta[-1,]), 2, sum)
    iname = "L1 Norm"
    approx.f = 1
  } else if (xvar == "lambda") { 
    index = log(x$lambda)
    iname = "Log Lambda"
    approx.f = 0
  } else {
    index = x$dev
    iname = "Fraction Deviance Explained"
    approx.f = 1
  }
  matplot(index, t(x$beta[-1,]), lty=1, xlab=iname, ylab="Coefficients", type="l", ...)
  atdf = pretty(index)
  prettydf = approx(x=index, y=x$df, xout=atdf, rule=2, method="constant", f=approx.f)$y
  axis(3, at=atdf, labels=prettydf, tcl=NA)
}

#' @title Prints pogsnet summary
#' @description Prings pogsnet summary
#' @param x pogsnet object
#' @param digits Number of digits to display
#' @param ... Not used.
#' @usage \method{print}{pogsnet}(x, digits = max(3, getOption("digits") - 3), ...)
#' @export
print.pogsnet <- function(x, digits = max(3, getOption("digits") - 3), ...){
  cat("\nCall: ", deparse(x$call), "\n\n")
  print(cbind(Df=x$df, "%Dev"=signif(x$dev.ratio, digits), Lambda=signif(x$lambda, digits)))
}

# Copy from glmnet
getmin <- function(lambda, cvm, cvsd) {
  cvmin = min(cvm, na.rm=TRUE)
  idmin = (cvm <= cvmin)
  lambda.min = max(lambda[idmin], na.rm=TRUE)
  idmin = match(lambda.min, lambda)
  semin = (cvm + cvsd)[idmin]
  idmin = (cvm <= semin)
  lambda.1se = max(lambda[idmin], na.rm=TRUE)
  list(lambda.min=lambda.min, lambda.1se=lambda.1se)
}

#' @title Cross-validation for glmnet
#' @description Does k-fold cross-validation for glmnet.
#' @param x Data matrix.
#' @param y Response vector
#' @param weights Weights of each training example, defaults to 1.
#' @param lambda List of lambda values for which CV should be performed.
#' @param nfolds Number of folds.
#' @param foldid Optional vector of values between 1 and nfolds, assigning
#' datapoint to cv-fold.
#' @param ... Other parameters to pogsnet.
#' @export
cv.pogsnet <- function(x, y, weights, lambda=NULL, nfolds=10, foldid, ...) {
  # Check input data
  if(!is.null(lambda) && length(lambda) < 2) {
    stop("Need more than one value of lambda for cv.pogsnet")
  }
  N = nrow(x)
  y = drop(y)
  if(missing(weights)) {
    weights = rep(1.0, N)
  } else {
    weights = as.double(weights)
  }
  if(missing(foldid)) {
    foldid = sample(rep(seq(nfolds), length=N))
  } else {
    nfolds = max(foldid)
  }
  if(nfolds < 3) {
    stop("nfolds must be bigger than 3; nfolds=10 recommended")
  }
  
  # Create initial pogsnet call (mostly for sequence of lambdas)
  pogsnet.call = match.call(expand.dots=TRUE)
  pogsnet.call[[1]] = as.name("pogsnet") 
  pogsnet.object = pogsnet(x, y, weights=weights, lambda=lambda, ...)
  pogsnet.object$call = pogsnet.call
  lambda = pogsnet.object$lambda

  # Containers for prediction result
  predmat = matrix(NA, length(y), length(lambda))
  nlams = double(nfolds)
  
  # Begin CV
  for(i in seq(nfolds)) {
    which = (foldid == i)
    if(is.matrix(y)) {
      y_sub = y[!which,] 
    } else {
      y_sub = y[!which]
    }
    pogsnet.fit = pogsnet(x[!which, , drop=FALSE], y_sub, lambda=lambda, weights=weights[!which], cutoff=FALSE, ...)
    nlami = length(pogsnet.fit$lambda)
    predmat[which, seq(nlami)] = predict(pogsnet.fit, x[which, , drop=FALSE], type="response")
  }
  N = length(y) - apply(is.na(predmat), 2, sum)
  
  # Process CV results
  cvraw = (y - predmat) ^ 2
  cvm = apply(cvraw, 2, weighted.mean, w=weights, na.rm=TRUE)
  cvsd = sqrt(apply(scale(cvraw, cvm, FALSE) ^ 2, 2, weighted.mean, w=weights, na.rm=TRUE) / (N - 1))
  
  # Prepare output
  out = list(lambda=lambda, cvm=cvm, cvsd=cvsd, cvup=cvm + cvsd, cvlo=cvm - cvsd,
            nzero=pogsnet.object$df, name="Mean-Squared Error", pogsnet.fit=pogsnet.object)
  obj = c(out,as.list(getmin(lambda, cvm, cvsd)))
  class(obj) = "cv.pogsnet"
  obj
}

# Copy from glmnet
error.bars <- function(x, upper, lower, width=0.02, ...) {
  xlim <- range(x)
  barw <- diff(xlim) * width
  segments(x, upper, x, lower, ...)
  segments(x - barw, upper, x + barw, upper, ...)
  segments(x - barw, lower, x + barw, lower, ...)
  range(upper, lower)
}

#' @title Plot pogsnet CV 
#' @description Plots results of pogsnet CV
#' @param x pogsnet.cv object
#' @param sign.lambda Either plot against log(lambda) (default) or its negative if sign.lambda=-1.
#' @param ... Other graphical parameters to plot.
#' @usage \method{plot}{cv.pogsnet}(x, sign.lambda, ...)
#' @export
plot.cv.pogsnet <- function(x, sign.lambda=1, ...) {
  # Set up plotting arguments
  cvobj = x
  xlab = "log(Lambda)"
  if(sign.lambda < 0) {
    xlab = paste("-", xlab, sep="")
  }
  plot.args = list(x=sign.lambda * log(cvobj$lambda), y=cvobj$cvm, ylim=range(cvobj$cvup, cvobj$cvlo), 
                   xlab=xlab, ylab=cvobj$name, type="n")
  new.args = list(...)
  if(length(new.args)) {
    plot.args[names(new.args)] = new.args
  }
  
  # Call plot function
  do.call("plot", plot.args)
  
  # Add error bars
  error.bars(sign.lambda * log(cvobj$lambda), cvobj$cvup, cvobj$cvlo, width=0.01, col="darkgrey")
  points(sign.lambda * log(cvobj$lambda), cvobj$cvm, pch=20, col="red")
  axis(side=3, at=sign.lambda * log(cvobj$lambda), labels=paste(cvobj$nz), tick=FALSE, line=0)
  abline(v=sign.lambda * log(cvobj$lambda.min), lty=3)
  abline(v=sign.lambda * log(cvobj$lambda.1se), lty=3)
  invisible()
}
