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