context("Test xgboost models")

source("helper-utils.R")

test_random_forest_classifier <- function(x, y) {
  model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)
  predictions <- model %>% predict(x)
  expect_equal(max(predictions), 1)
  expect_equal(min(predictions), 0)
  expect_true(model %>% score(x, y) > 0.90)
}

test_succeeds("Random Forest classifier works correctly with numeric labels", {
  x <- iris[1:4]
  y <- as.integer(iris$Species) - 1
  test_random_forest_classifier(x, y)
})

test_succeeds("Random Forest classifier works correctly with factor labels", {
  x <- iris[1:4]
  y <- iris$Species
  test_random_forest_classifier(x, y)
})
