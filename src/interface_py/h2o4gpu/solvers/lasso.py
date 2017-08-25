from h2o4gpu.solvers.elastic_net_base import GLM
"""
H2O Lasso Regression Solver

:param int n_threads: Number of threads to use in the gpu. Default is None.
:param int n_gpus: Number of gpu's to use in GLM solver. Default is -1.
:param bool fit_intercept: Include constant term in the model. Default is True.
:param int lambda_min_ratio: Minimum lambda used in lambda search. Default is 1e-7.
:param int n_lambdas: Number of lambdas to be used in a search. Default is 100.
:param int n_folds: Number of cross validation folds. Default is 1.
:param float tol: tolerance.  Default is 1E-2.
:param bool lambda_stop_early: Stop early when there is no more relative improvement on train or validation. Default is True.
:param bool glm_stop_early: Stop early when there is no more relative improvement in the primary and dual residuals for ADMM.  Default is True
:param float glm_stop_early_error_fraction: Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much). Default is 1.0.
:param int max_iter: Maximum number of iterations. Default is 5000
:param int verbose: Print verbose information to the console if set to > 0. Default is 0.
:param str family: Use "logistic" for classification with logistic regression. Defaults to "elasticnet" for regression. Must be "logistic" or "elasticnet".
:param lambda_max: Maximum Lambda value to use.  Default is None, and then internally compute standard maximum
"""
class Lasso(GLM):
    def __init__(
            self,
            n_threads=None,
            n_gpus=-1,
            fit_intercept=True,
            lambda_min_ratio=1E-7,
            n_lambdas=100,
            n_folds=1,
            tol=1E-2,
            lambda_stop_early=True,
            glm_stop_early=True,
            glm_stop_early_error_fraction=1.0,
            max_iter=5000,
            verbose=0,
            family="elasticnet",
            lambda_max=None,
    ):
        super(Lasso, self).__init__(
            n_threads=n_threads,
            n_gpus=n_gpus,
            intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=n_lambdas,
            n_folds=n_folds,
            n_alphas=1,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iterations=max_iter,
            verbose=verbose,
            family=family,
            give_full_path=0,
            lambda_max=lambda_max,
            alpha_max=1.0,
            alpha_min=1.0,
            order=None,)