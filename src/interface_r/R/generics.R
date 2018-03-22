#' Generic Method to Train an H2O4GPU Estimator
#' 
#' @param object The h2o4gpu model object
#' @param ... Additional arguments (unused for now).
#' @export
#' @rdname generics
#' @name fit
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @importFrom stats predict
#' @export
stats::predict

# Note that we are redefining `transform` generic method here to override `base::transform`'s signature.
#' Generic Method to Transform a Dataset using Trained H2O4GPU Estimator
#' 
#' @export
#' @rdname generics
#' @name transform
transform <- function(object, ...) {
  UseMethod("transform")
}
