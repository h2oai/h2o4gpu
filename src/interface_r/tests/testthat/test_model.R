context("Test h2o4gpu_model methods")

test_that("kmeans model can be constructed and trained", {
  np <- reticulate::import("numpy")
  
  x <- np$array(list(list(1, 1), list(1, 4), list(1, 0)))
  model <- h2o4gpu.kmeans(n_clusters = 2L, random_state = 1234L) %>% fit(x)
  
  predictions <- model %>% predict(x)
  transformed <- model %>% transform(x)
  model$model$cluster_centers_
})
