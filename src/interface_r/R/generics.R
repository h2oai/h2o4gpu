#' @export
h2o4gpu_model <- function(model, subclass = NULL) {
  structure(
    list(model = model),
    class = c(subclass, "h2o4gpu_model")
  )
}

#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @importFrom stats predict
#' @export
stats::predict

