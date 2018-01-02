# Util Functions

have_h2o4gpu <- function() {
  reticulate::py_module_available("h2o4gpu")
}

if (have_h2o4gpu()) {
  reticulate::use_python("/root/.pyenv/versions/3.6.1/bin/python", required = TRUE)
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