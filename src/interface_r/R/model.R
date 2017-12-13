#' @export
fit.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit(X = x, ...)
  object
}

#' @export
predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$predict(X = x, ...)
  object
}

#' @export
transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$transform(X = x, ...)
  object
}

#' @export
score.h2o4gpu_model <- function(object, x, ...) {
  object$model$score(X = x, ...)
  object
}

#' @export
fit_transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_transform(X = x, ...)
  object
}

#' @export
fit_predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_predict(X = x, ...)
  object
}
