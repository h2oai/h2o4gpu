# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# pylint: disable=unused-import
from h2o4gpu.solvers import elastic_net
from h2o4gpu.linear_model import base as sk
from ..solvers.utils import _setter


class LinearRegression(object):
    """H2O LinearRegression Regression Solver

    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
        an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, optional, default 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.

    tol : float, (Default=1E-2)
       Relative tolerance.

    n_gpus : int
        Number of gpu's to use in RandomForestRegressor solver. Default is -1.

    glm_stop_early : bool, (Default=True)
       Stop early when there is no more relative
       improvement in the primary and dual residuals for ADMM.

    glm_stop_early_error_fraction : float, (Default=1.0)
       Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at
       least this much).

    verbose : int, (Default=0)
       Print verbose information to the console if set to > 0.

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
            fit_intercept=True,  #h2o4gpu
            normalize=False,
            copy_X=True,
            n_jobs=1,
            n_gpus=-1,
            tol=1E-4,
            glm_stop_early=True,  # h2o4gpu
            glm_stop_early_error_fraction=1.0,  # h2o4gpu
            verbose=False,
            backend='auto',
            **kwargs):

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        self.do_daal = False
        self.do_sklearn = False

        if backend == 'auto':
            # Fall back to Sklearn
            # Can remove if fully implement sklearn functionality
            self.do_sklearn = False

            params_string = ['normalize']
            params = [normalize]
            params_default = [False]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + ". Will run Sklearn Linear Regression.")
                    self.do_sklearn = True
                i = i + 1
        elif backend == 'sklearn':
            self.do_sklearn = True
            self.backend = 'sklearn'
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
            self.backend = 'h2o4gpu'
        elif backend == 'daal':
            from h2o4gpu import DAAL_SUPPORTED
            if DAAL_SUPPORTED:
                from h2o4gpu.solvers.daal_solver.regression \
                    import LinearRegression as DLR
                self.do_daal = True
                self.backend = 'daal'

                self.model_daal = DLR(fit_intercept=fit_intercept,
                                      normalize=normalize,
                                      **kwargs)
            else:
                import platform
                print("WARNING:"
                      "DAAL is supported only for x86_64, "
                      "architecture detected {}. Sklearn model"
                      "used instead".format(platform.architecture()))
                self.do_sklearn = True
                self.backend = 'sklearn'

        self.model_sklearn = sk.LinearRegressionSklearn(
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
            lambda_max=lambda_max,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            alphas=alphas,
            lambdas=lambdas,
            tol_seek_factor=tol_seek_factor,
            family=family,
            order=None)

        if self.do_sklearn:
            if verbose:
                print("Running sklearn Linear Regression")
            self.model = self.model_sklearn
        elif self.do_daal:
            if verbose:
                print("Running PyDAAL Linear Regression")
            self.model = self.model_daal
        else:
            if verbose:
                print("Running h2o4gpu Linear Regression")
            self.model = self.model_h2o4gpu
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        if self.do_sklearn:
            res = self.model.fit(X, y, sample_weight)
            self.set_attributes()
        elif self.do_daal:
            res = self.model.fit(X, y)
        else:
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
        # TODO score for DAAL? input parameters are not clear
        if self.verbose:
            print("WARNING: score() is using sklearn")
        if not self.do_sklearn:
            self.model_sklearn.fit(X, y)  #Need to re-fit
        res = self.model_sklearn.score(X, y, sample_weight)
        return res

    def set_params(self, **params):
        return self.model.set_params(**params)

    def set_attributes(self):
        """ set attributes for Linear Regression
        """
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.coef_ = oself.model.coef_')
        s('oself.intercept_ = oself.model.intercept_')

        self.time_prepare = None
        s('oself.time_prepare = oself.model.time_prepare')
        self.time_upload_data = None
        s('oself.time_upload_data = oself.model.time_upload_data')
        self.time_fitonly = None
        s('oself.time_fitonly = oself.model.time_fitonly')
