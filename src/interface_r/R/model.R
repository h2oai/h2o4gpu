#' @export
h2o4gpu_model <- function(model, subclass = NULL, description = NULL) {
  r_model_obj <- structure(
    list(model = model, description = description),
    class = c(subclass, "h2o4gpu_model")
  )
  r_model_obj$params <- r_model_obj$model$get_params()
  r_model_obj
}

# Attrach attributes to R model object
attach_attrs_to_model <- function(r_model_obj) {
  attrs_exclude_from_attach <- c(
    "fit", "fit_predict", "fit_transform", "score", "predict", "transform",
    "init")
  attrs_exclude_from_attach <- c(attrs_exclude_from_attach, names(r_model_obj$params))
  
  if (grepl("H2O", as.character(r_model_obj$model))){
    model_attrs <- names(r_model_obj$model)
  } else {
    model_attrs <- names(r_model_obj$model$model)
  }
  # Attach attributes to the returned R model
  invisible(
    lapply(
      model_attrs[!model_attrs %in% attrs_exclude_from_attach],
      function(attrib) {
        if (grepl("H2O", as.character(r_model_obj$model))){
          r_model_obj[[attrib]] <<- r_model_obj$model[[attrib]]
        } else { 
          r_model_obj[[attrib]] <<- r_model_obj$model$model[[attrib]]
        }
      }))
  r_model_obj
}

#' @export
print.h2o4gpu_model <- function(x, ...) {
  # Note that the x is essentially the model object
  # The signature only contains x so it's consistent with the built=in print S3 generic method
  object <- x
  cat(paste0("Algorithm: ", object$description, "\n\n"))
  params <- object$params
  param_names <- names(params)
  param_items <- unlist(lapply(param_names, function(item) {
    if (is.null(params[[item]])) {
      item_value <- "NULL"
    } else if (params[[item]] == "") {
      item_value <- '""'
    } else if (is.character(params[[item]])) {
      item_value <- paste0('"', params[[item]], '"')
    } else {
      item_value <- params[[item]]
    }
    paste(item,
          item_value,
          collapse = "", sep = " = ")
  }))
  cat("Parameters:\n")
  cat(paste(param_items, collapse = ", "))
}

#' Train an H2O4GPU Estimator
#' 
#' This function builds the model using the training data specified.
#' 
#' @param object The h2o4gpu model object
#' @param x The training data where each column represents a different predictor variable 
#' to be used in building the model.
#' @param y A vector of numeric values to be used as response variable in building the model. 
#' Note that if the vector is character or factor, it will be converted to numeric column 
#' (e.g. 0, 1, 2, ...) implicitly. For unsupervised models, this argument can be ignored or
#' specified as `NULL`.
#' @param ... Additional arguments (unused for now).
#' 
#' @export
fit.h2o4gpu_model <- function(object, x, y = NULL, ...) {
  object$model$fit(X = resolve_model_input(x), y = resolve_model_y(y, class(object)), ...)
  attach_attrs_to_model(object)
}

#' Make Predictions using Trained H2O4GPU Estimator
#' 
#' This function makes predictions from new data using a trained H2O4GPU model and returns class predictions
#' for classification and predicted values for regression.
#' 
#' @param object The h2o4gpu model object
#' @param x The new data where each column represents a different predictor variable to 
#' be used in generating predictions.
#' @param type One of "raw" or "prob", indicating the type of output: predicted values or probabilities
#' @param ... Additional arguments (unused for now).
#' @export
predict.h2o4gpu_model <- function(object, x, type="raw", ...) {
  if (type == "raw") {
    preds <- object$model$predict(X = resolve_model_input(x), ...)
  } else if (type == "prob") {
    preds <- object$model$predict_proba(X = resolve_model_input(x), ...)
    if (!is.null(object$classes_)){
      colnames(preds) <- object$classes_ #Taken from tree based models
    }
  } else {
    stop(paste0("Unrecognized 'type' parameter value. Expected either 'raw' or 'prob but got ", type))
  }
  return(preds)
}

#' Transform a Dataset using Trained H2O4GPU Estimator
#' 
#' This function transforms the given new data using a trained H2O4GPU model.
#' 
#' @param object The h2o4gpu model object
#' @param x The new data where each column represents a different predictor variable to 
#' be used in generating predictions.
#' @param ... Additional arguments (unused for now).
#' @export
transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$transform(X = resolve_model_input(x), ...)
}
