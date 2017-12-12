#' @export
fit.h2o4gpu_model <- function(object, ...) {
  object$model$fit(...)
  object
}