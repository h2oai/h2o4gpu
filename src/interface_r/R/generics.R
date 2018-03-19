#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @importFrom stats predict
#' @export
stats::predict

# Note that we are redefining `transform` generic method here to override `base::transform`'s signature.
#' @export
transform <- function(object, ...) {
  UseMethod("transform")
}
