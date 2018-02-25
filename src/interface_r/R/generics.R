#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @importFrom stats predict
#' @export
stats::predict

#' @export
score <- function(object, ...) {
  UseMethod("score")
}

#' @export
fit_transform <- function(object, ...) {
  UseMethod("fit_transform")
}

#' @export
fit_predict <- function(object, ...) {
  UseMethod("fit_predict")
}

# Note that we are redefining `transform` generic method here to override `base::transform`'s signature.
#' @export
transform <- function(object, ...) {
  UseMethod("transform")
}
