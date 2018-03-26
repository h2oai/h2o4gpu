#' h2o4gpu in R
#' 
#' @docType package
#' @name h2o4gpu
#' 
#' @examples 
#' \dontrun{
#' 
#' library(h2o4gpu)
#' 
#' # Setup dataset
#' x <- iris[1:4]
#' y <- as.integer(iris$Species) - 1
#' 
#' # Initialize and train the classifier
#' model <- h2o4gpu.random_forest_classifier() %>% fit(x, y)
#' 
#' # Make predictions
#' predictions <- model %>% predict(x)
#' 
#' }
NULL

h2o4gpu <- NULL
np <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load handler
  displayed_warning <- FALSE
  
  delay_load <- list(
    
    priority = 5,
    
    environment = "r-h2o4gpu",
    
    on_load = function() {
      check_compatibility(displayed_warning)
    },
    
    on_error = function(e) {
      stop(e$error_message, call. = FALSE)
    }
  )
  
  h2o4gpu <<- reticulate::import("h2o4gpu", delay_load = delay_load)
  np <<- reticulate::import("numpy", convert = FALSE, delay_load = TRUE)
}

# Placeholder for now
check_compatibility <- function(displayed_warning) {

}
