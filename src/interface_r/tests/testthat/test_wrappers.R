context("Test model wrappers")

source("helper-utils.R")

classifier_name <- "h2o4gpu.random_forest_classifier"
classifier <- h2o4gpu.random_forest_classifier

test_succeeds(paste0("Test ", classifier_name), {
  dataset <- simple_dataset("classification")
  x <- dataset$x
  y <- dataset$y
  model <- classifier %>% fit(dataset$x, dataset$y)
  predictions <- model %>% predict(x)
  expect_equal(max(predictions), 1)
  expect_equal(min(predictions), 0)
  expect_true(model %>% score(x, y) > 0.90)
})
