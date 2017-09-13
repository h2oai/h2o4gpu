#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.solvers.elastic_net import GLM


class LinearRegression(GLM):
    """H2O Linear Regression Solver

    :param int n_threads: Number of threads to use in the gpu. Default is None.

    :param int n_gpus: Number of gpu's to use in GLM solver. Default is -1.

    :param bool fit_intercept: Include constant term in the model.
        Default is True.

    :param int n_folds: Number of cross validation folds. Default is 1.

    :param float tol: tolerance.  Default is 1E-2.

    :param float tol_seek_factor : factor of tolerance to seek
        once below null model accuracy.  Default is 1E-1, so seeks tolerance
        of 1E-3 once below null model accuracy for tol=1E-2.

    :param bool glm_stop_early: Stop early when there is no more relative
        improvement in the primary and dual residuals for ADMM.
        Default is True.

    :param float glm_stop_early_error_fraction: Relative tolerance for
        metric-based stopping criterion (stop if relative improvement is not
        at least this much). Default is 1.0.

    :param int max_iter: Maximum number of iterations. Default is 5000.

    :param int verbose: Print verbose information to the console if set to > 0.
        Default is 0.

    :param int store_full_path : Extract full regularization path from glm model
    """

    def __init__(self,
                 n_threads=None,
                 n_gpus=-1,
                 fit_intercept=True,
                 n_folds=1,
                 tol=1E-2,
                 tol_seek_factor=1E-1,
                 glm_stop_early=True,
                 glm_stop_early_error_fraction=1.0,
                 max_iter=5000,
                 verbose=0,
                 store_full_path=0):
        super(LinearRegression, self).__init__(
            n_threads=n_threads,
            n_gpus=n_gpus,
            fit_intercept=fit_intercept,
            lambda_min_ratio=0.0,
            n_lambdas=1,
            n_folds=n_folds,
            n_alphas=1,
            tol=tol,
            tol_seek_factor=tol_seek_factor,
            lambda_stop_early=False,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
            family='elasticnet',
            lambda_max=0.0,
            alpha_max=0.0,
            alpha_min=0.0,
            store_full_path=store_full_path,
            order=None)
