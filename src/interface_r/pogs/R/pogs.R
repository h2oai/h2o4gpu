#' @title Proximal Operator Graph Solver
#' @description Solver for convex optimization problems in the form
#' \deqn{\min. f(y) + g(x), \text{ s.t. } y = Ax}{min. f(y) + g(x),  s.t. y = Ax,}
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
#' @param params List of parameters (rel_tol=1e-4, abs_tol=1e-4, rho=1.0,
#' max_iter=1000, quiet=FALSE). All parameters are optional and take on a
#' default value if not specified.
#' @examples
#' # Specify Lasso problem.
#' A = matrix(rnorm(100 * 10), 100, 10)
#' b = rnorm(100)
#' lambda = 5
#' f = list(h = kSquare(), b = b)
#' g = list(h = kAbs(), c = lambda)
#' pogs(A, f, g)
#' @useDynLib pogs
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

  # Check that f$h and g$h exist.
  if (!any(names(f) == "h")) {
    stop("pogs(): field f$h is required!")
  }
  if (!any(names(g) == "h")) {
    stop("pogs(): field g$h is required!")
  }

  # Check fields in f.
  for (name in names(f)) {
    if (!any(name == c("a", "b", "c", "d", "e", "h"))) {
      stop(cat("pogs(): field f$", name, " unknown!", sep=""))
    }
    if (!is.vector(f[[name]])) {
      stop(cat("pogs(): field f$", name, " must be a vector!", sep=""))
    }
    if (!is.numeric(f[[name]])) {
      stop(cat("pogs(): field f$", name, " must be numeric!", sep=""))
    }
    if (length(f[[name]]) != nrow(A) && length(f[[name]]) != 1) {
      stop(cat("pogs(): field f$", name, " must must have length nrow(A) or 1!", sep=""))
    }
    f[[name]] = as.double(f[[name]])
  }

  # Check fields in g.
  for (name in names(g)) {
    if (!any(name == c("a", "b", "c", "d", "e", "h"))) {
      stop(cat("pogs(): field g$", name, " unknown!", sep=""))
    }
    if (!is.vector(g[[name]])) {
      stop(cat("pogs(): field g$", name, " must be a vector!", sep=""))
    }
    if (!is.numeric(g[[name]])) {
      stop(cat("pogs(): field g$", name, " must be numeric!", sep=""))
    }
    if (length(g[[name]]) != ncol(A) && length(g[[name]]) != 1) {
      stop(cat("pogs(): field g$", name, " must must have length ncol(A) or 1!", sep=""))
    }
    g[[name]] = as.double(g[[name]])
  }
 
  # Check fields in params.
  for (name in names(params)) {
    if (!any(name == c("rel_tol", "abs_tol", "rho", "max_iter", "quiet"))) {
      stop(cat("pogs(): field params$", name, " unknown!", sep=""))
    }
    if (!is.numeric(params[[name]]) && name != "quiet") {
      stop(cat("pogs(): field params$", name, " must be numeric!", sep=""))
    }
    if (!is.logical(params[[name]]) && name == "quiet") {
      stop(cat("pogs(): field params$", name, " must be logical!", sep=""))
    }
    if (length(params[[name]]) != 1) {
      stop(cat("pogs(): field params$", name, " must must have length 1!", sep=""))
    }
  }
  
  # Call POGS.
  .Call("PogsWrapper", A, f, g, params)
}

#' @title Absolute Value Function
#' @description \deqn{h(x) = |x|}
#' @export
kAbs <- function(m=1) {
  rep(0, m)
}

#' @title Exponential Function
#' @description \deqn{h(x) = e^x}
#' @export
kExp <- function(m=1) {
  rep(1, m)
}

#' @title Huber Function
#' @description \deqn{h(x) = huber(x)}
#' @export
kHuber <- function(m=1) {
  rep(2, m)
}

#' @title Identity Function
#' @description \deqn{h(x) = x}
#' @export
kIdentity <- function(m=1) {
  rep(3, m)
}

#' @title Indicator Function of [0, 1] Box
#' @description \deqn{h(x) = I(0 <= x <= 1)}
#' @export
kIndBox01 <- function(m=1) {
  rep(4, m)
}

#' @title Indicator Function of Origin
#' @description \deqn{h(x) = I(x = 0)}
#' @export
kIndEq0 <- function(m=1) {
  rep(5, m)
}

#' @title Indicator Function of Non-Negative Orthant
#' @description \deqn{h(x) = I(x >= 0)}
#' @export
kIndGe0 <- function(m=1) {
  rep(6, m)
}

#' @title Indicator Function of Non-Positive Orthant
#' @description \deqn{h(x) = I(x <= 0)}
#' @export
kIndLe0 <- function(m=1) {
  rep(7, m)
}

#' @title Logistic Function
#' @description \deqn{h(x) = log(1 + e^x)}
#' @export
kLogistic <- function(m=1) {
  rep(8, m)
}

#' @title Positive-Part Function
#' @description \deqn{h(x) = max(0, x)}
#' @export
kMaxPos0 <- function(m=1) {
  rep(9, m)
}

#' @title Negative-Part Function
#' @description \deqn{h(x) = max(0, -x)}
#' @export
kMaxNeg0 <- function(m=1) {
  rep(10, m)
}

#' @title Negative Entropy Function
#' @description \deqn{h(x) = x log(x)}
#' @export
kNegEntr <- function(m=1) {
  rep(11, m)
}

#' @title Negative Log Function
#' @description \deqn{h(x) = -log(x)}
#' @export
kNegLog <- function(m=1) {
  rep(12, m)
}

#' @title Reciprocal Function
#' @description \deqn{h(x) = 1 / x}
#' @export
kRecipr <- function(m=1) {
  rep(13, m)
}

#' @title Square Function
#' @description \deqn{h(x) = (1/2) x^2}
#' @export
kSquare <- function(m=1) {
  rep(14, m)
}

#' @title Zero Function
#' @description \deqn{h(x) = 0}
#' @export
kZero <- function(m=1) {
  rep(15, m)
}
