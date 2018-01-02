context("Test h2o4gpu_model methods")

reticulate::use_virtualenv("h2o4gpu_env")
np <- reticulate::import("numpy")

test_generic_methods <- function(input) {
  model <- h2o4gpu.kmeans(n_clusters = 2L, random_state = 1234L) %>% fit(input)
  expect_equal(model$cluster_centers_, structure(c(1, 1, 0.5, 4), .Dim = c(2L, 2L)))
  expect_equal(model %>% predict(input), structure(c(0L, 1L, 0L), .Dim = 3L))
  expect_equal(model %>% transform(input), structure(c(0.5, 3.5, 0.5, 3, 0, 4), .Dim = c(3L, 2L)))
  expect_equal(model %>% score(input), -0.5)
}

test_that("Generic methods for model work correctly for different types of inputs", {
  raw_input <- list(list(1, 1), list(1, 4), list(1, 0))
  x <- np$array(raw_input)
  
  # Test support for data.frame
  test_generic_methods(as.data.frame(x))
  # Test support for list
  test_generic_methods(raw_input)
  # Test support for matrix/np.array
  test_generic_methods(x)
})
