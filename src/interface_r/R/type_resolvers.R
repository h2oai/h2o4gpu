as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

resolve_model_input <- function(x) {
  if (is.matrix(x)) {
    x
  } else if (is.data.frame(x)) {
    as.matrix(x)
  } else {
    stop(paste0("Input x of type: ", class(x), " is not currently supported."))
  }
}
