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
    stop(paste0('Input x of type "', class(x), '" is not currently supported.'))
  }
}

resolve_model_y <- function(y, cls) {
  if (is.null(y)) {
    NULL
  } else {
    # Implicitly convert character and factor column to numeric (for classifiers only)
    if ("classifier" %in% cls && (is.character(y) || is.factor(y))) {
      warning('Your model input "y" is either character or factor. ',
              'It will be converted to numeric column [0, 1, 2, ...] implicitly.')
      y <- as.integer(as.factor(y)) - 1
    } else {
      if ("regressor" %in% cls && !is.numeric(y)) {
        stop('Your model input "y" is non-numeric. Please convert it to numeric. ')
      }
    }
    resolve_model_input(y)
  }
}
