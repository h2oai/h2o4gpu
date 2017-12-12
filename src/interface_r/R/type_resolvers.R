as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}

