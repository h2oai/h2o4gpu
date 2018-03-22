#' h2o4gpu in R
#' 
#' @docType package
#' @name h2o4gpu
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
  
  if (reticulate::py_module_available("h2o4gpu") && reticulate::py_module_available("np")) {
    h2o4gpu <<- reticulate::import("h2o4gpu", delay_load = delay_load)
    np <<- reticulate::import("numpy", convert = FALSE, delay_load = TRUE)
  }
}

# Placeholder for now
check_compatibility <- function(displayed_warning) {

}
