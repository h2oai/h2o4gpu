# Utility function to generate the R wrappers for different models
gen_wrapper <- function(
  python_function,
  r_function = NULL,
  additional_int_params = NULL,
  nullable_int_params = NULL) {

  docs <- reticulate::py_function_docs(python_function)
  con <- textConnection("wrapper", "w")

  write("#' @export", file = con)
  if (is.null(r_function)) {
    r_function <- docs$name
  }

  # Generate function signature
  signature <- sub(paste0(docs$name, "\\("),
                   paste(r_function, "<- function(\n\t"), docs$signature)
  signature <- gsub(', ', ',\n\t', signature)
  write(paste(signature, "{\n"), file = con)
  
  # Execution of Python API
  write(paste0("  model <- ", python_function, "("), file = con)
  
  params <- names(docs$parameters)
  # Extract the params with integer types
  int_params <- gsub(" = [-]?[0-9]+L", "", stringr::str_extract_all(signature, "[A-z]+ = [-]?[0-9]+L")[[1]])
  if (!is.null(additional_int_params)) int_params <- c(int_params, additional_int_params)

  # Generate parameters that get passed to the Python API call
  if (length(params) > 0) {
    for (i in 1:length(params)) {
      param <- params[i]
      # Custom handling of integer params
      if (param %in% int_params) {
        param <- paste0("as.integer(", param, ")")
      }
      # Custom handling of nullable integer params
      if (!is.null(nullable_int_params) && param %in% nullable_int_params) {
        param <- paste0("as_nullable_integer(", param, ")")
      }
      suffix <- ifelse(i < length(params), ",", "\n  )")
      write(paste0("    ", params[[i]], " = ", param, 
                   suffix), file = con)
    }
  } else {
    write(")", file = con)
  }

  # Attach additional class information
  if (grepl("classifier", python_function, ignore.case = TRUE)) {
    class_tags <- 'c("classifier")'
  } else if (grepl("regressor", python_function, ignore.case = TRUE)) {
    class_tags <- 'c("regressor")'
  } else {
    class_tags <- 'NULL'
  }
  write(paste0('  h2o4gpu_model(model, ', class_tags, ')'), file = con)

  write("}\n", file = con)
  close(con)
  class(wrapper) <- c("py_wrapper", "character")
  wrapper
}
