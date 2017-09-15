# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import coordinate_descent as sk
from ..solvers.utils import _setter

class Lasso(object):
    """H2O Lasso Regression Solver

        Selects between h2o4gpu.solvers.elastic_net.GLM
        and h2o4gpu.linear_model.coordinate_descent.Lasso_sklearn
        Documentation:
        import h2o4gpu.solvers ; help(h2o4gpu.solvers.elastic_net.GLM)
        help(h2o4gpu.linear_model.lasso.Lasso_sklearn)
    """

    def __init__(self,
                 alpha=1.0, #h2o4gpu
                 fit_intercept=True, #h2o4gpu
                 normalize=False,
                 precompute=False,
                 copy_X=True,
                 max_iter=5000, #h2o4gpu
                 tol=1e-2, #h2o4gpu
                 warm_start=False,
                 positive=False,
                 random_state=None,
                 selection='cyclic',
                 n_gpus=-1,  # h2o4gpu
                 glm_stop_early=True,  # h2o4gpu
                 glm_stop_early_error_fraction=1.0, #h2o4gpu
                 verbose=False): # h2o4gpu

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False

        params_string = ['normalize', 'precompute', 'copy_X',
                         'warm_start', 'positive', 'random_state',
                         'selection']
        params = [normalize, precompute, copy_X,
                  warm_start, positive, random_state,
                  selection]
        params_default = [False, False, True, False, False, None, 'cyclic']

        i = 0
        self.do_sklearn = False
        for param in params:
            if param != params_default[i]:
                self.do_sklearn = True
                print("WARNING: The sklearn parameter " + params_string[i]
                      + " has been changed from default to "
                      + str(param) + ". Will run Sklearn Lasso Regression.")
                self.do_sklearn = True
            i = i + 1

        self.model_sklearn = sk.Lasso_sklearn(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection)
        n_threads = None
        n_alphas = 1
        n_lambdas = 1
        n_folds = 1
        lambda_max = None
        lambda_min_ratio = 1.0
        lambda_stop_early = False
        store_full_path = 1
        alphas = None
        lambdas = None
        alpha_min = alpha
        alpha_max = alpha

        self.model_h2o4gpu = elastic_net.GLM(
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
            store_full_path=store_full_path,
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            order=None)

        if self.do_sklearn:
            print("Running sklearn Lasso Regression")
            self.model = self.model_sklearn
        else:
            print("Running h2o4gpu Lasso Regression")
            self.model = self.model_h2o4gpu

    def fit(self, X, y=None, check_input=True):
        if self.do_sklearn:
            res = self.model.fit(X, y, check_input)
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
        s('oself.sparse_coef_ = oself.model.sparse_coef_')
        s('oself.intercept_ = oself.model.intercept_')
        s('oself.n_iter_ = oself.model.n_iter_')
