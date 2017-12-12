#' Install h2o4gpu from a path to wheel file
#' @export
install_h2o4gpu <- function(wheel_path) {
  # TODO: Support other install methods
  conda_env <- "r-h2o4gpu"
  # Remove this once PR is merged to reticulate
  conda_path <- "/home/terry/miniconda3/bin/conda"
  if (!conda_env %in% reticulate::conda_list(conda = conda_path)$name) {
    conda_create(envname = conda_env, conda = conda_path)
  } else {
    conda_remove(envname = conda_env, conda = conda_path)
    conda_create(envname = conda_env, conda = conda_path)
  }
  use_condaenv(conda_env, conda = conda_path)
  conda_install(envname = conda_env, conda = conda_path, packages = wheel_path, pip = TRUE, pip_ignore_installed = TRUE)
}
