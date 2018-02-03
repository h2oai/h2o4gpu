# Utility function to generate the R wrappers for different models
gen_wrapper <- function(
  python_function,
  r_function = NULL,
  additional_int_params = NULL,
  nullable_int_params = NULL,
  class_tags = NULL) {

  write_line <- function(text) {
    cat(text, sep = "\n")
  }

  capture.output(
    {
      docs <- reticulate::py_function_docs(python_function)
      
      write_line("#' @export")
      if (is.null(r_function)) {
        r_function <- docs$name
      }
      
      # Generate function signature
      signature <- sub(paste0(docs$name, "\\("),
                       paste(r_function, "<- function(\n\t"), docs$signature)
      signature <- gsub('backend = "auto"', 'backend = "h2o4gpu"', signature)
      signature <- gsub('family = "elasticnet",', "", signature)
      signature <- gsub(', ', ',\n\t', signature)
      write_line(paste(signature, "{\n"))
      
      # Execution of Python API
      write_line(paste0("  model <- ", python_function, "("))
      
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
          if (param == "family" && r_function %in% c("h2o4gpu.elastic_net_classifier","h2o4gpu.elastic_net_regressor")){
            if(r_function == "h2o4gpu.elastic_net_classifier"){
              write_line(paste0("    ", params[[i]], " = ", '"logistic"', suffix))
            } else {
              write_line(paste0("    ", params[[i]], " = ", '"elasticnet"', suffix))
            }
          } else {
            write_line(paste0("    ", params[[i]], " = ", param, suffix))
          }
        }
      } else {
        write_line(")")
      }
      
      # Attach additional class information
      write_line(paste0('  h2o4gpu_model(model, ', class_tags, ')'))
      
      write_line("}\n")
    }
  )
}

# Append the wrapper 
write_wrapper <- function(python_function,
                          file_name,
                          r_function = NULL,
                          additional_int_params = NULL,
                          nullable_int_params = NULL,
                          class_tags = NULL) {
  write(
    gen_wrapper(
      python_function,
      r_function,
      additional_int_params,
      nullable_int_params,
      class_tags
      ),
    file = file_name,
    append = TRUE)
}
