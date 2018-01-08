as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

resolve_model_input <- function(x) {
  if (is.matrix(x) || is.numeric(x)) {
    x
  } else if (is.data.frame(x)) {
    as.matrix(x)
  } else if (is.list(x)) {
    np$array(x)
  } else {
    stop(paste0("Input x of type \"", class(x), "\" is not currently supported."))
  }
}

resolve_model_y <- function(y) {
  if (is.null(y)) {
    NULL
  } else {
    resolve_model_input(y)
  }
}
