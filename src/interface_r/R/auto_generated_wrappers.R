# -------------------------------------------------------------------------------
#             *** WARNING: DO NOT MODIFY THIS FILE ***
#              
# Instead, modify scripts/gen_wrappers.R which automatically generates this file
# ------------------------------------------------------------------------------- 

#' @export
h2o4gpu.random_forest_classifier <- function(
	n_estimators = 10L,
	criterion = "gini",
	max_depth = 3L,
	min_samples_split = 2L,
	min_samples_leaf = 1L,
	min_weight_fraction_leaf = 0.0,
	max_features = "auto",
	max_leaf_nodes = NULL,
	min_impurity_decrease = 0.0,
	min_impurity_split = NULL,
	bootstrap = TRUE,
	oob_score = FALSE,
	n_jobs = 1L,
	random_state = NULL,
	verbose = 0L,
	warm_start = FALSE,
	class_weight = NULL,
	subsample = 1.0,
	colsample_bytree = 1.0,
	num_parallel_tree = 1L,
	tree_method = "gpu_hist",
	n_gpus = -1L,
	predictor = "gpu_predictor",
	backend = "h2o4gpu") {

  model <- h2o4gpu$RandomForestClassifier(
    n_estimators = as.integer(n_estimators),
    criterion = criterion,
    max_depth = as.integer(max_depth),
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = min_weight_fraction_leaf,
    max_features = max_features,
    max_leaf_nodes = max_leaf_nodes,
    min_impurity_decrease = min_impurity_decrease,
    min_impurity_split = min_impurity_split,
    bootstrap = bootstrap,
    oob_score = oob_score,
    n_jobs = as.integer(n_jobs),
    random_state = as_nullable_integer(random_state),
    verbose = as.integer(verbose),
    warm_start = warm_start,
    class_weight = class_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    num_parallel_tree = as.integer(num_parallel_tree),
    tree_method = tree_method,
    n_gpus = as.integer(n_gpus),
    predictor = predictor,
    backend = backend
  )
  h2o4gpu_model(model, c("classifier"), "Random Forest Classifier")
}

#' @export
h2o4gpu.random_forest_regressor <- function(
	n_estimators = 10L,
	criterion = "mse",
	max_depth = 3L,
	min_samples_split = 2L,
	min_samples_leaf = 1L,
	min_weight_fraction_leaf = 0.0,
	max_features = "auto",
	max_leaf_nodes = NULL,
	min_impurity_decrease = 0.0,
	min_impurity_split = NULL,
	bootstrap = TRUE,
	oob_score = FALSE,
	n_jobs = 1L,
	random_state = NULL,
	verbose = 0L,
	warm_start = FALSE,
	subsample = 1.0,
	colsample_bytree = 1.0,
	num_parallel_tree = 1L,
	tree_method = "gpu_hist",
	n_gpus = -1L,
	predictor = "gpu_predictor",
	backend = "h2o4gpu") {

  model <- h2o4gpu$RandomForestRegressor(
    n_estimators = as.integer(n_estimators),
    criterion = criterion,
    max_depth = as.integer(max_depth),
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = min_weight_fraction_leaf,
    max_features = max_features,
    max_leaf_nodes = max_leaf_nodes,
    min_impurity_decrease = min_impurity_decrease,
    min_impurity_split = min_impurity_split,
    bootstrap = bootstrap,
    oob_score = oob_score,
    n_jobs = as.integer(n_jobs),
    random_state = as_nullable_integer(random_state),
    verbose = as.integer(verbose),
    warm_start = warm_start,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    num_parallel_tree = as.integer(num_parallel_tree),
    tree_method = tree_method,
    n_gpus = as.integer(n_gpus),
    predictor = predictor,
    backend = backend
  )
  h2o4gpu_model(model, c("regressor"), "Random Forest Regressor")
}

#' @export
h2o4gpu.gradient_boosting_classifier <- function(
	loss = "deviance",
	learning_rate = 0.1,
	n_estimators = 100L,
	subsample = 1.0,
	criterion = "friedman_mse",
	min_samples_split = 2L,
	min_samples_leaf = 1L,
	min_weight_fraction_leaf = 0.0,
	max_depth = 3L,
	min_impurity_decrease = 0.0,
	min_impurity_split = NULL,
	init = NULL,
	random_state = NULL,
	max_features = "auto",
	verbose = 0L,
	max_leaf_nodes = NULL,
	warm_start = FALSE,
	presort = "auto",
	colsample_bytree = 1.0,
	num_parallel_tree = 1L,
	tree_method = "gpu_hist",
	n_gpus = -1L,
	predictor = "gpu_predictor",
	backend = "h2o4gpu") {

  model <- h2o4gpu$GradientBoostingClassifier(
    loss = loss,
    learning_rate = learning_rate,
    n_estimators = as.integer(n_estimators),
    subsample = subsample,
    criterion = criterion,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = min_weight_fraction_leaf,
    max_depth = as.integer(max_depth),
    min_impurity_decrease = min_impurity_decrease,
    min_impurity_split = min_impurity_split,
    init = init,
    random_state = as_nullable_integer(random_state),
    max_features = max_features,
    verbose = as.integer(verbose),
    max_leaf_nodes = max_leaf_nodes,
    warm_start = warm_start,
    presort = presort,
    colsample_bytree = colsample_bytree,
    num_parallel_tree = as.integer(num_parallel_tree),
    tree_method = tree_method,
    n_gpus = as.integer(n_gpus),
    predictor = predictor,
    backend = backend
  )
  h2o4gpu_model(model, c("classifier"), "Gradient Boosting Classifier")
}

#' @export
h2o4gpu.gradient_boosting_regressor <- function(
	loss = "ls",
	learning_rate = 0.1,
	n_estimators = 100L,
	subsample = 1.0,
	criterion = "friedman_mse",
	min_samples_split = 2L,
	min_samples_leaf = 1L,
	min_weight_fraction_leaf = 0.0,
	max_depth = 3L,
	min_impurity_decrease = 0.0,
	min_impurity_split = NULL,
	init = NULL,
	random_state = NULL,
	max_features = "auto",
	alpha = 0.9,
	verbose = 0L,
	max_leaf_nodes = NULL,
	warm_start = FALSE,
	presort = "auto",
	colsample_bytree = 1.0,
	num_parallel_tree = 1L,
	tree_method = "gpu_hist",
	n_gpus = -1L,
	predictor = "gpu_predictor",
	backend = "h2o4gpu") {

  model <- h2o4gpu$GradientBoostingRegressor(
    loss = loss,
    learning_rate = learning_rate,
    n_estimators = as.integer(n_estimators),
    subsample = subsample,
    criterion = criterion,
    min_samples_split = as.integer(min_samples_split),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_weight_fraction_leaf = min_weight_fraction_leaf,
    max_depth = as.integer(max_depth),
    min_impurity_decrease = min_impurity_decrease,
    min_impurity_split = min_impurity_split,
    init = init,
    random_state = as_nullable_integer(random_state),
    max_features = max_features,
    alpha = alpha,
    verbose = as.integer(verbose),
    max_leaf_nodes = max_leaf_nodes,
    warm_start = warm_start,
    presort = presort,
    colsample_bytree = colsample_bytree,
    num_parallel_tree = as.integer(num_parallel_tree),
    tree_method = tree_method,
    n_gpus = as.integer(n_gpus),
    predictor = predictor,
    backend = backend
  )
  h2o4gpu_model(model, c("regressor"), "Gradient Boosting Regressor")
}

#' @export
h2o4gpu.elastic_net_regressor <- function(
	alpha = 1.0,
	l1_ratio = 0.5,
	fit_intercept = TRUE,
	normalize = FALSE,
	precompute = FALSE,
	max_iter = 5000L,
	copy_X = TRUE,
	tol = 0.01,
	warm_start = FALSE,
	positive = FALSE,
	random_state = NULL,
	selection = "cyclic",
	n_gpus = -1L,
	lambda_stop_early = TRUE,
	glm_stop_early = TRUE,
	glm_stop_early_error_fraction = 1.0,
	verbose = FALSE,
	n_threads = NULL,
	gpu_id = 0L,
	lambda_min_ratio = 1e-07,
	n_lambdas = 100L,
	n_folds = 5L,
	n_alphas = 5L,
	tol_seek_factor = 0.1,
	 store_full_path = 0L,
	lambda_max = NULL,
	alpha_max = 1.0,
	alpha_min = 0.0,
	alphas = NULL,
	lambdas = NULL,
	double_precision = NULL,
	order = NULL,
	backend = "h2o4gpu") {

  model <- h2o4gpu$ElasticNet(
    alpha = alpha,
    l1_ratio = l1_ratio,
    fit_intercept = fit_intercept,
    normalize = normalize,
    precompute = precompute,
    max_iter = as.integer(max_iter),
    copy_X = copy_X,
    tol = tol,
    warm_start = warm_start,
    positive = positive,
    random_state = as_nullable_integer(random_state),
    selection = selection,
    n_gpus = as.integer(n_gpus),
    lambda_stop_early = lambda_stop_early,
    glm_stop_early = glm_stop_early,
    glm_stop_early_error_fraction = glm_stop_early_error_fraction,
    verbose = verbose,
    n_threads = n_threads,
    gpu_id = as.integer(gpu_id),
    lambda_min_ratio = lambda_min_ratio,
    n_lambdas = as.integer(n_lambdas),
    n_folds = as.integer(n_folds),
    n_alphas = as.integer(n_alphas),
    tol_seek_factor = tol_seek_factor,
    family = "elasticnet",
    store_full_path = as.integer(store_full_path),
    lambda_max = lambda_max,
    alpha_max = alpha_max,
    alpha_min = alpha_min,
    alphas = alphas,
    lambdas = lambdas,
    double_precision = double_precision,
    order = order,
    backend = backend
  )
  h2o4gpu_model(model, c("regressor"), "Elastic Net Regressor")
}

#' @export
h2o4gpu.elastic_net_classifier <- function(
	alpha = 1.0,
	l1_ratio = 0.5,
	fit_intercept = TRUE,
	normalize = FALSE,
	precompute = FALSE,
	max_iter = 5000L,
	copy_X = TRUE,
	tol = 0.01,
	warm_start = FALSE,
	positive = FALSE,
	random_state = NULL,
	selection = "cyclic",
	n_gpus = -1L,
	lambda_stop_early = TRUE,
	glm_stop_early = TRUE,
	glm_stop_early_error_fraction = 1.0,
	verbose = FALSE,
	n_threads = NULL,
	gpu_id = 0L,
	lambda_min_ratio = 1e-07,
	n_lambdas = 100L,
	n_folds = 5L,
	n_alphas = 5L,
	tol_seek_factor = 0.1,
	 store_full_path = 0L,
	lambda_max = NULL,
	alpha_max = 1.0,
	alpha_min = 0.0,
	alphas = NULL,
	lambdas = NULL,
	double_precision = NULL,
	order = NULL,
	backend = "h2o4gpu") {

  model <- h2o4gpu$ElasticNet(
    alpha = alpha,
    l1_ratio = l1_ratio,
    fit_intercept = fit_intercept,
    normalize = normalize,
    precompute = precompute,
    max_iter = as.integer(max_iter),
    copy_X = copy_X,
    tol = tol,
    warm_start = warm_start,
    positive = positive,
    random_state = as_nullable_integer(random_state),
    selection = selection,
    n_gpus = as.integer(n_gpus),
    lambda_stop_early = lambda_stop_early,
    glm_stop_early = glm_stop_early,
    glm_stop_early_error_fraction = glm_stop_early_error_fraction,
    verbose = verbose,
    n_threads = n_threads,
    gpu_id = as.integer(gpu_id),
    lambda_min_ratio = lambda_min_ratio,
    n_lambdas = as.integer(n_lambdas),
    n_folds = as.integer(n_folds),
    n_alphas = as.integer(n_alphas),
    tol_seek_factor = tol_seek_factor,
    family = "logistic",
    store_full_path = as.integer(store_full_path),
    lambda_max = lambda_max,
    alpha_max = alpha_max,
    alpha_min = alpha_min,
    alphas = alphas,
    lambdas = lambdas,
    double_precision = double_precision,
    order = order,
    backend = backend
  )
  h2o4gpu_model(model, c("classifier"), "Elastic Net Classifier")
}

#' @export
h2o4gpu.kmeans <- function(
	n_clusters = 8L,
	init = "k-means++",
	n_init = 1L,
	max_iter = 300L,
	tol = 0.0001,
	precompute_distances = "auto",
	verbose = 0L,
	random_state = NULL,
	copy_x = TRUE,
	n_jobs = 1L,
	algorithm = "auto",
	gpu_id = 0L,
	n_gpus = -1L,
	do_checks = 1L,
	backend = "h2o4gpu") {

  model <- h2o4gpu$KMeans(
    n_clusters = as.integer(n_clusters),
    init = init,
    n_init = as.integer(n_init),
    max_iter = as.integer(max_iter),
    tol = tol,
    precompute_distances = precompute_distances,
    verbose = as.integer(verbose),
    random_state = as_nullable_integer(random_state),
    copy_x = copy_x,
    n_jobs = as.integer(n_jobs),
    algorithm = algorithm,
    gpu_id = as.integer(gpu_id),
    n_gpus = as.integer(n_gpus),
    do_checks = as.integer(do_checks),
    backend = backend
  )
  h2o4gpu_model(model, NULL, "KMeans Clustering")
}

#' @export
h2o4gpu.pca <- function(
	n_components = 2L,
	copy = TRUE,
	whiten = FALSE,
	svd_solver = "arpack",
	tol = 0.0,
	iterated_power = "auto",
	random_state = NULL,
	verbose = FALSE,
	backend = "h2o4gpu") {

  model <- h2o4gpu$PCA(
    n_components = as.integer(n_components),
    copy = copy,
    whiten = whiten,
    svd_solver = svd_solver,
    tol = tol,
    iterated_power = iterated_power,
    random_state = as_nullable_integer(random_state),
    verbose = verbose,
    backend = backend
  )
  h2o4gpu_model(model, NULL, "Principal Components Analysis (PCA)")
}

#' @export
h2o4gpu.truncated_svd <- function(
	n_components = 2L,
	algorithm = "power",
	n_iter = 100L,
	random_state = NULL,
	tol = 1e-05,
	verbose = FALSE,
	backend = "h2o4gpu",
	n_gpus = 1L,
	gpu_id = 0L) {

  model <- h2o4gpu$TruncatedSVD(
    n_components = as.integer(n_components),
    algorithm = algorithm,
    n_iter = as.integer(n_iter),
    random_state = as_nullable_integer(random_state),
    tol = tol,
    verbose = verbose,
    backend = backend,
    n_gpus = as.integer(n_gpus),
    gpu_id = as.integer(gpu_id)
  )
  h2o4gpu_model(model, NULL, "Truncated Singular Value Decomposition (TruncatedSVD)")
}

