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
