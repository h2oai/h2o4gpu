#' @export
h2o4gpu_model <- function(model, subclass = NULL) {
  r_model_obj <- structure(
    list(model = model),
    class = c(subclass, "h2o4gpu_model")
  )
  r_model_obj
}

# Attrach attributes to R model object
attach_attrs_to_model <- function(r_model_obj) {
  attrs_exclude_from_attach <- c(
    "fit", "fit_predict", "fit_transform", "score", "predict", "transform",
    "init")
  model_attrs <- names(r_model_obj$model$model)
  # Attach attributes to the returned R model
  invisible(
    lapply(
      model_attrs[!model_attrs %in% attrs_exclude_from_attach],
      function(attrib) {
        r_model_obj[[attrib]] <<- r_model_obj$model$model[[attrib]]
      }))
  r_model_obj
}

#' @export
fit.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit(X = resolve_model_input(x), ...)
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
score.h2o4gpu_model <- function(object, x, ...) {
  object$model$score(X = resolve_model_input(x), ...)
}

#' @export
fit_transform.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_transform(X = resolve_model_input(x), ...)
}

#' @export
fit_predict.h2o4gpu_model <- function(object, x, ...) {
  object$model$fit_predict(X = resolve_model_input(x), ...)
}
