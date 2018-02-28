# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import print_function
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import ridge as sk
from h2o4gpu.solvers.daal_solver.regression import RidgeRegression as DRR
from ..solvers.utils import _setter


class Ridge(object):
    """H2O Ridge Regression Solver

        Selects between h2o4gpu.solvers.elastic_net.GLM
        and h2o4gpu.linear_model.ridge.Ridge_sklearn
        Documentation:
        import h2o4gpu.solvers ; help(h2o4gpu.solvers.elastic_net.ElasticNetH2O)
        help(h2o4gpu.linear_model.ridge.Ridge_sklearn)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
            alpha=1.0,  #h2o4gpu
            fit_intercept=True,  #h2o4gpu
            normalize=False,
            copy_X=True,
            max_iter=5000,  #h2o4gpu
            tol=1e-2,  #h2o4gpu
            solver='auto',
            random_state=None,
            n_gpus=-1,  # h2o4gpu
            glm_stop_early=True,  # h2o4gpu
            glm_stop_early_error_fraction=1.0,  #h2o4gpu
            verbose=False,
            backend='auto',
            **kwargs):  # h2o4gpu

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        self.do_daal = False
        self.do_sklearn = False
        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':
            params_string = ['normalize', 'solver']
            params = [normalize, solver]
            params_default = [False, 'auto']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + ". Will run Sklearn Ridge Regression.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
            self.backend = 'sklearn'
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
            self.backend = 'h2o4gpu'
        elif backend == 'daal':
            self.do_daal = True
            self.backend = 'daal'

        self.model_daal = DRR(fit_intercept=fit_intercept,
                              normalize=normalize,
                              **kwargs)

        self.model_sklearn = sk.RidgeSklearn(
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state)

        # Equivalent Ridge parameters for h2o4gpu
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
        alpha_min = 0.0
        alpha_max = 0.0

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
                print("Running sklearn Ridge Regression")
            self.model = self.model_sklearn
        elif self.do_daal:
            if verbose:
                print("Running PyDAAL Ridge Regression")
            self.model = self.model_daal
        else:
            if verbose:
                print("Running h2o4gpu Ridge Regression")
            self.model = self.model_h2o4gpu
        self.verbose = verbose

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
        if self.verbose:
            print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  #Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """ set attributes for Ridge
        """
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.coef_ = oself.model.coef_')
        s('oself.intercept_ = oself.model.intercept_')
        s('oself.n_iter_ = oself.model.n_iter_')

        self.time_prepare = None
        s('oself.time_prepare = oself.model.time_prepare')
        self.time_upload_data = None
        s('oself.time_upload_data = oself.model.time_upload_data')
        self.time_fitonly = None
        s('oself.time_fitonly = oself.model.time_fitonly')
