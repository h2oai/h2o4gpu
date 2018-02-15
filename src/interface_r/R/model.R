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

#' @export
fit.h2o4gpu_model <- function(object, x, y = NULL, ...) {
  object$model$fit(X = resolve_model_input(x), y = resolve_model_y(y, class(object)), ...)
  attach_attrs_to_model(object)
}

#' @export
predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$predict(X = resolve_model_input(x), ...)
}

#' @export
transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$transform(X = resolve_model_input(x), ...)
}

#' @export
score.h2o4gpu_model <- function(object, x, y = NULL, ...) {
  object$model$score(X = resolve_model_input(x), y = resolve_model_y(y, class(object)), ...)
}

#' @export
fit_transform.h2o4gpu_model <- function(object, x, y = NULL, ...) {
  object$model$fit_transform(X = resolve_model_input(x), y = resolve_model_y(y, class(object)), ...)
}

#' @export
fit_predict.h2o4gpu_model <- function(object, x, y = NULL, ...) {
  object$model$fit_predict(X = resolve_model_input(x), y = resolve_model_y(y, class(object)), ...)
}
