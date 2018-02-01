context("Test h2o4gpu_model methods")

source("helper-utils.R")

test_generic_methods <- function(input) {
  model <- h2o4gpu.kmeans(n_clusters = 2, random_state = 1234) %>% fit(input)
  expect_equal(dim(model$cluster_centers_), c(2, 2))
  expect_equal(dim(model %>% predict(input)), 3)
  expect_equal(dim(model %>% transform(input)), c(3, 2))
}

test_succeeds("Generic methods for model work correctly for different types of inputs", {
  raw_input <- list(list(1, 1), list(1, 4), list(1, 0))
  x <- np$array(raw_input)

  # Test support for data.frame
  test_generic_methods(as.data.frame(x))
  # Test support for list
  test_generic_methods(raw_input)
  # Test support for matrix/np.array
  test_generic_methods(x)
})
