context("Test xgboost models")

source("helper-utils.R")

test_random_forest_classifier <- function(x, y) {
  # Suppress intentional warnings
  suppressWarnings(
    {
      model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)
      predictions <- model %>% predict(x)
      expect_equal(length(y), length(predictions))
    }
  )
}

test_succeeds("Random Forest classifier works correctly with numeric labels", {
  x <- iris[1:4]
  y <- as.integer(iris$Species) - 1
  test_random_forest_classifier(x, y)
})

test_succeeds("Random Forest classifier works correctly with sparse feature matrix", {
  if (require("Matrix")) {
    N <- 100
    x <- sparseMatrix(
      i = sample(N, N),
      j = sample(N, N),
      x = runif(N),
      dims = c(N, N))
    y <- c(rep(0, N/2), rep(1, N/2))
    test_random_forest_classifier(x, y)
  }
})

test_succeeds("Random Forest classifier works correctly with factor labels", {
  x <- iris[1:4]
  y <- iris$Species
  test_random_forest_classifier(x, y)
})

test_succeeds("Random Forest regressor works correctly", {
  x <- longley[1:6]
  y <- longley$Employed
  model <- h2o4gpu.random_forest_regressor() %>% fit(x, y)
  predictions <- model %>% predict(x)
  expect_true(0 < predictions && predictions < 100)
  expect_error(h2o4gpu.random_forest_regressor() %>% fit(x, as.character(y)), regexp = 'Your model input "y" is non-numeric')
})
