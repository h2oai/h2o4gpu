#' h2o4gpu in R
#' 
#' @docType package
#' @name h2o4gpu
NULL

h2o4gpu <- NULL
np <- NULL

.onLoad <- function(libname, pkgname) {
  
  search_env <- "r-h2o4gpu"
  # delay load handler
  displayed_warning <- FALSE
  
  delay_load <- list(
    
    priority = 5,
    
    environment = search_env,
    
    on_load = function() {
      check_compatibility(displayed_warning)
      if (search_env %in% reticulate::conda_list()$name) {
        # TODO: Support other load methods (future work)
        use_condaenv(search_env)
      } else {
        message("h2o4gpu Python package is not installed. ",
                "Please use install_h2o4gpu(wheel_path) to install. ",
                "You can find suitable Python wheel for your environment here: ",
                "https://github.com/h2oai/h2o4gpu#installation.")
      }
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
