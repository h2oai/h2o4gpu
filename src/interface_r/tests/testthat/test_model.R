context("Test h2o4gpu_model methods")

test_that("kmeans model works correctly", {
  np <- reticulate::import("numpy")
  
  x <- np$array(list(list(1, 1), list(1, 4), list(1, 0)))
  model <- h2o4gpu.kmeans(n_clusters = 2L, random_state = 1234L) %>% fit(x)
  expect_equal(model$model$cluster_centers_, structure(c(1, 1, 0.5, 4), .Dim = c(2L, 2L)))
  
  # Test other generic methods
  expect_equal(model %>% predict(x), structure(c(0L, 1L, 0L), .Dim = 3L))
  expect_equal(model %>% transform(x), structure(c(0.5, 3.5, 0.5, 3, 0, 4), .Dim = c(3L, 2L)))
  expect_equal(model %>% score(x), -0.5)
  
  # Test support for data.frame
  model <- h2o4gpu.kmeans(n_clusters = 2L, random_state = 1234L) %>% fit(as.data.frame(x))
})
