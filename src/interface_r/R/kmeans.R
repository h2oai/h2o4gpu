#' K-Means clustering
#' 
#' @export
h2o4gpu.kmeans <- function(
  n_clusters = 8L,
  init = 'k-means++',
  n_init = 1L,
  max_iter = 300L,
  tol = 1e-4,
  precompute_distances = 'auto',
  verbose = 0,
  random_state = NULL,
  copy_x = TRUE,
  n_jobs = 1,
  algorithm = 'auto',
  gpu_id = 0,
  n_gpus = -1,
  do_checks = 1
) {
  model <- h2o4gpu$KMeans(
    n_clusters = as.integer(n_clusters),
    init = init,
    n_init = as.integer(n_init),
    max_iter = as.integer(max_iter),
    tol = as.integer(tol),
    precompute_distances = precompute_distances,
    verbose = as.integer(verbose),
    random_state = as_nullable_integer(random_state),
    copy_x = copy_x,
    n_jobs = as.integer(n_jobs),
    algorithm = algorithm,
    gpu_id = as.integer(gpu_id),
    n_gpus = as.integer(n_gpus),
    do_checks = as.integer(do_checks)
  )
  h2o4gpu_model(model, "kmeans")
}
