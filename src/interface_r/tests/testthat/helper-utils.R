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
    skip_on_cran()
    skip_if_no_h2o4gpu()
    expect_error(force(expr), NA)
  })
}

simple_dataset <- function(type = c("unsupervised", "classification", "regression")) {
  if (type == "unsupervised") {
    x <- structure(c(1, 1, 1, 1, 4, 0, 6, 7, 8, 9, 1, 2, 2, 3, 4, 5), .Dim = c(8L, 2L))
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

test_classifier <- function(classifier_func, classifier_name) {
  test_succeeds(paste0("Test ", classifier_name), {
    dataset <- simple_dataset("classification")
    x <- dataset$x
    y <- dataset$y
    model <- classifier_func() %>% fit(x, y)
    predictions <- model %>% predict(x)
    expect_equal(max(predictions), 2)
    expect_equal(min(predictions), 0)
  })
}

test_regressor <- function(regressor_func, regressor_name) {
  test_succeeds(paste0("Test ", regressor_name), {
    dataset <- simple_dataset("regression")
    x <- dataset$x
    y <- dataset$y
    model <- regressor_func() %>% fit(x, y)
    predictions <- model %>% predict(x)
    expect_true(0 < predictions && predictions < 100)
  })
}

test_unsupervised <- function(model_func, model_name) {
  test_succeeds(paste0("Test ", model_name), {
    dataset <- simple_dataset("unsupervised")
    x <- dataset$x
    model <- model_func() %>% fit(x)
    expect_equal(dim(model %>% predict(x)), nrow(x))
    expect_equal(nrow(model %>% transform(x)), nrow(x))
  })
}
