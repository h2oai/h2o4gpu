# -*- encoding: utf-8 -*-
"""
:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from h2o4gpu.solvers.elastic_net import GLM


class LogisticRegression(GLM):
    """H2O Logistic Regression Solver

    :param int n_threads: Number of threads to use in the gpu. Default is None.
    :param int n_gpus: Number of gpu's to use in GLM solver. Default is -1.
    :param bool fit_intercept: Include constant term in the model.
        Default is True.
    :param float lambda_min_ratio: Minimum lambda ratio to maximum lambda, used
        in lambda search.
        Default is 1e-7.
    :param int n_lambdas: Number of lambdas to be used in a search.
        Default is 100.
    :param int n_folds: Number of cross validation folds. Default is 1.
    :param int n_alphas: Number of alphas to be used in a search.
        Default is 1.
    :param float tol: tolerance.  Default is 1E-2.
    :param bool lambda_stop_early: Stop early when there is no more
        relative improvement on train or validation. Default is True.
    :param bool glm_stop_early: Stop early when there is no more relative
        improvement in the primary and dual residuals for ADMM. Default is True
    :param float glm_stop_early_error_fraction: Relative tolerance for
        metric-based stopping criterion (stop if relative improvement is not at
        least this much). Default is 1.0.
    :param int max_iter: Maximum number of iterations. Default is 5000
    :param int verbose: Print verbose information to the console if set to > 0.
        Default is 0.
    :param lambda_max: Maximum Lambda value to use.  Default is None, and then
        internally compute standard maximum
    :param alpha_max: Maximum alpha. Default is 1.0.
    :param alpha_min: Minimum alpha. Default is 0.0.
    """
    def __init__(
            self,
            n_threads=None,
            n_gpus=-1,
            fit_intercept=True,
            lambda_min_ratio=1E-7,
            n_lambdas=100,
            n_folds=1,
            n_alphas=1,
            tol=1E-2,
            lambda_stop_early=True,
            glm_stop_early=True,
            glm_stop_early_error_fraction=1.0,
            max_iter=5000,
            verbose=0,
            give_full_path=0,
            lambda_max=None,
            alpha_max=1.0,
            alpha_min=0.0,
    ):
        super(LogisticRegression, self).__init__(
            n_threads=n_threads,
            n_gpus=n_gpus,
            fit_intercept=fit_intercept,
            lambda_min_ratio=lambda_min_ratio,
            n_lambdas=n_lambdas,
            n_folds=n_folds,
            n_alphas=n_alphas,
            tol=tol,
            lambda_stop_early=lambda_stop_early,
            glm_stop_early=glm_stop_early,
            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
            max_iter=max_iter,
            verbose=verbose,
            family='logistic',
            give_full_path=give_full_path,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            order=None)
