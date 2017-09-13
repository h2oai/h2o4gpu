#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.solvers.elastic_net import GLM


class Lasso(GLM):
    """H2O Lasso Regression Solver

    :param int n_threads: Number of threads to use in the gpu. Default is None.

    :param int n_gpus: Number of gpu's to use in GLM solver. Default is -1.

    :param bool fit_intercept: Include constant term in the model.
        Default is True.

    :param float lambda_min_ratio: Minimum lambda used in lambda search.
        Default is 1e-7.

    :param int n_lambdas: Number of lambdas to be used in a search.
        Default is 100.

    :param int n_folds: Number of cross validation folds.
        Default is 1.

    :param float tol: tolerance.  Default is 1E-2.

    :param float tol_seek_factor : factor of tolerance to seek
        once below null model accuracy.  Default is 1E-1, so seeks tolerance
        of 1E-3 once below null model accuracy for tol=1E-2.

    :param bool lambda_stop_early: Stop early when there is no more relative
        improvement on train or validation. Default is True.

    :param bool glm_stop_early: Stop early when there is no more relative
        improvement in the primary and dual residuals for ADMM. Default is True.

    :param float glm_stop_early_error_fraction: Relative tolerance for
        metric-based stopping criterion (stop if relative improvement is not at
        least this much). Default is 1.0.

    :param int max_iter: Maximum number of iterations. Default is 5000.

    :param int verbose: Print verbose information to the console if set to > 0.
        Default is 0.

    :param int store_full_path : Extract full regularization path from glm model

    :param str family: Use "logistic" for classification with logistic
        regression. Defaults to "elasticnet" for regression.
        Must be "logistic" or "elasticnet".

    :param lambda_max: Maximum Lambda value to use.  Default is None, and then
        internally compute standard maximum.

    :param int,float lambdas: list, tuple, array, or numpy 1D array of lambdas,
        overrides n_lambdas, lambda_max, and lambda_min_ratio. Default is None.
    """

    def __init__(self,
                 n_threads=None,
                 n_gpus=-1,
                 fit_intercept=True,
                 lambda_min_ratio=1E-7,
                 n_lambdas=100,
                 n_folds=1,
                 tol=1E-2,
                 tol_seek_factor=1E-1,
                 lambda_stop_early=True,
                 glm_stop_early=True,
                 glm_stop_early_error_fraction=1.0,
                 max_iter=5000,
                 verbose=0,
                 family="elasticnet",
                 store_full_path=0,
                 lambda_max=None,
                 lambdas=None):
        super(Lasso, self).__init__(
            n_threads=n_threads,
            n_gpus=n_gpus,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=n_lambdas,
            n_folds=n_folds,
            n_alphas=1,
            tol=tol,
            tol_seek_factor=tol_seek_factor,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
            family=family,
            store_full_path=store_full_path,
            lambda_max=lambda_max,
            alpha_max=1.0,
            alpha_min=1.0,
            lambdas=lambdas,
            order=None)
