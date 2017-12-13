#' @export
fit.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit(X = resolve_model_input(x), ...)
  object
}

#' @export
predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$predict(X = resolve_model_input(x), ...)
  object
}

#' @export
transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$transform(X = resolve_model_input(x), ...)
  object
}

#' @export
score.h2o4gpu_model <- function(object, x, ...) {
  object$model$score(X = resolve_model_input(x), ...)
  object
}

#' @export
fit_transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_transform(X = resolve_model_input(x), ...)
  object
}

#' @export
fit_predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_predict(X = resolve_model_input(x), ...)
  object
}
