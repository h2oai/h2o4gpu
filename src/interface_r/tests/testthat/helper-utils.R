# Util Functions

have_h2o4gpu <- function() {
  reticulate::py_module_available("h2o4gpu")
}

if (have_h2o4gpu()) {
  # Currently pyenv is only available inside docker
  py_env_binary <- "/root/.pyenv/versions/3.6.1/bin/python"
  if (file.exists(py_env_binary)) {
    reticulate::use_python(py_env_binary, required = TRUE)
  }
  np <<- reticulate::import("numpy")
}

skip_if_no_h2o4gpu <- function() {
  if (!have_h2o4gpu())
    skip("h2o4gpu not available for testing")
}

test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_h2o4gpu()
    expect_error(force(expr), NA)
  })
}

simple_dataset <- function(type = c("unsupervised", "classification", "regression")) {
  if (type == "unsupervised") {
    x <- structure(c(1, 1, 1, 1, 4, 0), .Dim = c(3L, 2L))
    y <- NULL
  } else if (type == "classification") {
    x <- iris[1:4]
    y <- as.integer(iris$Species) - 1
  } else {
    x <- longley[1:6]
    y <- longley$Employed
  }
  return(
    list(
      x = x,
      y = y
    ))
}
