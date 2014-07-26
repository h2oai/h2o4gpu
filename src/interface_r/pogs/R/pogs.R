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
#' max_iter=1000, quiet=FALSE, adaptive_rho=TRUE). 
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
    stop("params must be a named list (elements abs_tol, rel_tol, rho, max_iter, quiet, adaptive_rho)")
  }
  for (name in names(params)) {
    if (!any(name == c("rel_tol", "abs_tol", "rho", "max_iter", "quiet", "adaptive_rho"))) {
      stop(cat("pogs(): field params$", name, " unknown!", sep=""))
    }
    if (!is.numeric(params[[name]]) && name != "quiet" && name != "adaptive_rho") {
      stop(cat("pogs(): field params$", name, " must be numeric!", sep=""))
    }
    if (!is.logical(params[[name]]) && (name == "quiet" || name == "adaptive_rho")) {
      stop(cat("pogs(): field params$", name, " must be logical!", sep=""))
    }
    if (length(params[[name]]) != 1) {
      stop(cat("pogs(): field params$", name, " must must have length 1!", sep=""))
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
#' @export
pogsnet <- function(x, y, family=c("gaussian", "binomial"),
                    weights, alpha=1, nlambda=100,
                    lambda.min.ratio=ifelse(nobs < nvars, 0.01, 0.0001), lambda=NULL,
                    penalty.factor=rep(1, nvars), intercept=TRUE, params=list(quiet=TRUE)) {

  # Check Family and Call
  family = match.arg(family)
  this.call = match.call()
  
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
  if (nlambda > 1) {
    for (i in 2:nlambda) {
      if (max(abs(soln$x[, i] - soln$x[, i - 1])) < 1e-4 * sum(abs(soln$x[, i]))) {
        last = i
        break
      }
    }
  }
  fit$beta = Matrix(soln$x[,1:last], sparse=TRUE)
  fit$df = apply(soln$x[,1:last], 2, function(x) { sum(x != 0) })
  fit$lambda = lambda[1:last]
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
  type=match.arg(type)
  if (is.null(s)) {
    beta = fit$beta
  } else {
    lamlist = lambda.interp(fit$lambda, s)
    beta = fit$beta[, lamlist$left, drop=FALSE] %*% Diagonal(x=lamlist$frac) + 
      fit$beta[, lamlist$right, drop=FALSE] %*% Diagonal(x=1 - lamlist$frac)
  }
  fitted = as.matrix(cbind2(1, newx) %*% beta)
  if (any(class(fit.po) == "binomial")) {
    if (type == "response") {
      fitted = 1 / (1 + exp(-lin))
    } else if (type == "class") {
      fitted = 1 / (1 + exp(-lin)) > 1/2
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
    coefs = fit$beta
  } else {
    lamlist = lambda.interp(fit$lambda, s)
    coefs = fit$beta[, lamlist$left, drop=FALSE] %*% Diagonal(x=lamlist$frac) + 
      fit$beta[, lamlist$right, drop=FALSE] %*% Diagonal(x=1 - lamlist$frac)
  }
  rownames(coefs) = c("(Intercept)", paste("V",seq(1, nrow(coefs) - 1), sep=""))
  coefs
}


