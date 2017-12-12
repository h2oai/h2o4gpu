#' K-Means clustering
#' 
#' @export
KMeans <- function(...) {
  model <- h2o4gpu$KMeans(...)
  h2o4gpu_model(model, "KMeans")
}
