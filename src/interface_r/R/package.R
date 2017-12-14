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
      search_env <- "r-h2o4gpu"
      conda_path <- "/home/terry/miniconda3/bin/conda"
      if (search_env %in% reticulate::conda_list(conda = conda_path)$name) {
        use_condaenv(search_env, conda = conda_path)
      } else {
        message("h2o4gpu Python package is not installed. Please use install_h2o4gpu(path_to_wheel) to install.")
      }
    },
    
    on_error = function(e) {
      stop(e$error_message, call. = FALSE)
    }
  )
  
  h2o4gpu <<- reticulate::import("h2o4gpu", delay_load = delay_load)
  np <<- import("numpy", convert = FALSE, delay_load = TRUE)
}

# Placeholder for now
check_compatibility <- function(displayed_warning) {

}
