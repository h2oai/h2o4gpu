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

        Selects between h2o4gpu.solvers.elastic_net.ElasticNetH2O
        and h2o4gpu.linear_model.coordinate_descent.Lasso_sklearn
        Documentation:
        import h2o4gpu.solvers ; help(h2o4gpu.solvers.elastic_net.ElasticNetH2O)
        help(h2o4gpu.linear_model.lasso.Lasso_sklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
            alpha=1.0,  #h2o4gpu
            fit_intercept=True,  #h2o4gpu
            normalize=False,
            precompute=False,
            copy_X=True,
            max_iter=5000,  #h2o4gpu
            tol=1e-2,  #h2o4gpu
            warm_start=False,
            positive=False,
            random_state=None,
            selection='cyclic',
            n_gpus=-1,  # h2o4gpu
            glm_stop_early=True,  # h2o4gpu
            glm_stop_early_error_fraction=1.0,  #h2o4gpu
            verbose=False,
            backend='auto'):  # h2o4gpu

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':
            params_string = ['normalize', 'positive', 'selection']
            params = [normalize, positive, selection]
            params_default = [False, False, 'cyclic']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + ". Will run Sklearn Lasso Regression.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        if self.do_sklearn:
            self.backend = 'sklearn'
        else:
            self.backend = 'h2o4gpu'

        self.model_sklearn = sk.LassoSklearn(
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

        #Equivalent Lasso parameters for h2o4gpu
        n_threads = None
        n_alphas = 1
        n_lambdas = 1
        n_folds = 1
        lambda_max = alpha
        lambda_min_ratio = 1.0
        lambda_stop_early = False
        store_full_path = 1
        alphas = None
        lambdas = None
        alpha_min = 1.0
        alpha_max = 1.0

        self.model_h2o4gpu = elastic_net.ElasticNetH2O(
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
            if verbose:
                print("Running sklearn Lasso Regression")
            self.model = self.model_sklearn
        else:
            if verbose:
                print("Running h2o4gpu Lasso Regression")
            self.model = self.model_h2o4gpu
        self.verbose = verbose

    def fit(self, X, y=None, check_input=True):
        """H2O Lasso Regression Fitter
        """

        if self.do_sklearn:
            res = self.model.fit(X, y, check_input)
            self.set_attributes()
            return res
        import numpy as np
        # FIXME: only works if numpy input
        if len(X.shape) == 2:
            sample_weight = X[:, 0] * 0.0 + 1.0 / (2.0 * np.shape(X)[0])
        else:
            sample_weight = X[:] * 0.0 + 1.0 / (2.0 * np.shape(X)[0])
        res = self.model.fit(X, y, sample_weight=sample_weight)
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
        if self.verbose:
            print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  #Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """ set attributes for Lasso
        """
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.coef_ = oself.model.coef_')
        s('oself.sparse_coef_ = oself.model.sparse_coef_')
        s('oself.intercept_ = oself.model.intercept_')
        s('oself.n_iter_ = oself.model.n_iter_')

        self.time_prepare = None
        s('oself.time_prepare = oself.model.time_prepare')
        self.time_upload_data = None
        s('oself.time_upload_data = oself.model.time_upload_data')
        self.time_fitonly = None
        s('oself.time_fitonly = oself.model.time_fitonly')
