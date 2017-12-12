context("Test h2o4gpu_model methods")

test_that("KMeans model can be constructed and trained", {
  require(reticulate)
  use_condaenv("py36", conda = "/home/terry/miniconda3/envs/py36/bin/conda")
  
  h2o4gpu <- import("h2o4gpu")
  np <- import("numpy")
  
  x <- np$array(list(list(1, 1), list(1, 4), list(1, 0)))
  model <- KMeans(n_clusters = 2L, random_state = 1234L) %>% fit(x)
  model$model$cluster_centers_
})
