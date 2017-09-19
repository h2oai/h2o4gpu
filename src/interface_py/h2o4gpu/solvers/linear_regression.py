# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import base as sk
from ..solvers.utils import _setter

class LinearRegression(object):
    """H2O LinearRegression Regression Solver

        Selects between h2o4gpu.solvers.elastic_net.ElasticNet_h2o4gpu
        and h2o4gpu.linear_model.base.LinearRegression_sklearn
        Documentation:
        import h2o4gpu.solvers ; help(h2o4gpu.solvers.elastic_net.ElasticNet_h2o4gpu)
        help(h2o4gpu.linear_model.base.LinearRegression_sklearn)
    """

    def __init__(self,
                 fit_intercept=True,  #h2o4gpu
                 normalize=False,
                 copy_X=True,
                 n_jobs=1,
                 n_gpus=-1,
                 tol=1E-4,
                 glm_stop_early=True,  # h2o4gpu
                 glm_stop_early_error_fraction=1.0,  # h2o4gpu
                 verbose=False):

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False

        params_string = ['normalize', 'copy_X', 'n_jobs']
        params = [normalize, copy_X, n_jobs]
        params_default = [False, True, 1]

        i = 0
        self.do_sklearn = False
        for param in params:
            if param != params_default[i]:
                self.do_sklearn = True
                print("WARNING: The sklearn parameter " + params_string[i]
                      + " has been changed from default to "
                      + str(param) + ". Will run Sklearn Linear Regression.")
                self.do_sklearn = True
            i = i + 1

        self.model_sklearn = sk.LinearRegression_sklearn(
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            n_jobs=n_jobs)

        # Equivalent Linear Regression parameters for h2o4gpu
        n_threads = None
        n_gpus = n_gpus
        fit_intercept = fit_intercept
        lambda_min_ratio = 0.0
        n_lambdas = 1
        n_folds = 1
        n_alphas = 1
        tol = tol
        tol_seek_factor = 1E-1
        lambda_stop_early = False
        glm_stop_early = glm_stop_early
        glm_stop_early_error_fraction = glm_stop_early_error_fraction
        max_iter = 5000
        verbose = verbose
        family = 'elasticnet'
        lambda_max = 0.0
        alpha_max = 0.0
        alpha_min = 0.0
        alphas = None
        lambdas = None

        self.model_h2o4gpu = elastic_net.ElasticNet_h2o4gpu(
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
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            tol_seek_factor=tol_seek_factor,
            family=family,
            order=None)

        if self.do_sklearn:
            print("Running sklearn Linear Regression")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu Linear Regression")
            self.model = self.model_h2o4gpu

    def fit(self, X, y=None, sample_weight=None):
        if self.do_sklearn:
            res = self.model.fit(X, y, sample_weight)
            self.set_attributes()
            return res
        res = self.model.fit(X, y)
        self.set_attributes()
        return res

    def get_params(self):
        return self.model.get_params()

    def predict(self, X):
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def score(self, X, y, sample_weight=None):
        # TODO add for h2o4gpu
        print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y) #Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.coef_ = oself.model.coef_')
        s('oself.intercept_ = oself.model.intercept_')
