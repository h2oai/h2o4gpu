#' Install h2o4gpu from a path to wheel file
#' @export
install_h2o4gpu <- function(wheel_path) {
  # TODO: Support other install methods (future work)
  conda_env <- "r-h2o4gpu"
  if (!conda_env %in% reticulate::conda_list()$name) {
    conda_create(envname = conda_env)
  } else {
    conda_remove(envname = conda_env)
    conda_create(envname = conda_env)
  }
  use_condaenv(conda_env)
  conda_install(envname = conda_env, packages = wheel_path, pip = TRUE, pip_ignore_installed = TRUE)
}
