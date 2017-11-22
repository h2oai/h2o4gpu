# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np
from ..libs.lib_pca import parameters
from ..solvers.utils import _setter
from ..solvers.truncated_svd import TruncatedSVDH2O, TruncatedSVD, _as_fptr


class PCAH2O(TruncatedSVDH2O):
    """Principal Component Analysis (PCA)

    Dimensionality reduction using truncated Singular Value Decomposition for GPU

    This implementation uses the ARPACK implementation of the truncated SVD.
    Contrary to SVD, this estimator does center the data before computing
    the singular value decomposition.

    :param: n_components Desired dimensionality of output data

    :param: whiten : bool, optional
        When True (False by default) the `components_` vectors are multiplied
        by the square root of (n_samples) and divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
    """

    def __init__(self, n_components=2, whiten=False):
        super().__init__(n_components)
        self.whiten = whiten
        self.n_components_ = n_components

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """Fit PCA on matrix X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :param y Ignored, for ScikitLearn compatibility

        :returns self : object

        """
        self.fit_transform(X)
        return self

    # pylint: disable=unused-argument
    def fit_transform(self, X, y=None):
        """Fit PCA on matrix X and perform dimensionality reduction
           on X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :param: y Ignored, for ScikitLearn compatibility

        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always be a
                         dense array.

        """
        X = np.asfortranarray(X, dtype=np.float64)
        Q = np.empty((self.n_components, X.shape[1]),
                     dtype=np.float64, order='F')
        U = np.empty((X.shape[0], self.n_components),
                     dtype=np.float64, order='F')
        w = np.empty(self.n_components, dtype=np.float64)
        explained_variance = np.empty(self.n_components,
                                      dtype=np.float64)
        explained_variance_ratio = np.empty(self.n_components,
                                            dtype=np.float64)
        param = parameters()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.whiten = self.whiten

        lib = self._load_lib()
        lib.pca(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U),
                          _as_fptr(explained_variance),
                          _as_fptr(explained_variance_ratio), param)

        # TODO mean_ and noise_variance_ calculation
        # can be done inside lib.pca if a bottleneck
        self.mean_ = np.mean(X, axis=0)
        n_samples, n_features = X.shape
        total_var = np.var(X, ddof=1, axis=0)
        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = \
                (total_var.sum() - self.explained_variance_.sum())
            self.noise_variance_ /= \
                min(n_features, n_samples) - self.n_components
        else:
            self.noise_variance_ = 0.

        self._Q = Q
        self._w = w
        self._U = U
        self._X = X
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

        X_transformed = U * w
        return X_transformed

    @property
    def mean_(self):
        return self.mean_

    @property
    def n_components_(self):
        return self.n_components_

    @property
    def noise_variance_(self):
        return self.noise_variance_

    # Util to load gpu lib
    def _load_lib(self):
        from ..libs.lib_pca import GPUlib

        gpu_lib = GPUlib().get()

        return gpu_lib


class PCA(TruncatedSVD):
    """
        PCA Wrapper

        Selects between h2o4gpu.decomposition.PCASklearn
        and h2o4gpu.solvers.pca.PCAH2O

        Documentation:
        import h2o4gpu.decomposition ;
        help(h2o4gpu.decomposition.PCASklearn)
        help(h2o4gpu.solvers.pca.PCA)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    # pylint: disable=unused-argument
    def __init__(self,
                 n_components=2,
                 copy=True,
                 whiten=False,
                 svd_solver="arpack",
                 tol=0.,
                 iterated_power="auto",
                 random_state=None,
                 verbose=False,
                 backend='auto'):
        super().__init__(n_components, random_state, tol, verbose, backend)
        self.svd_solver = svd_solver
        self.whiten = whiten

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':
            params_string = [
                'svd_solver', 'random_state', 'tol', 'iterated_power'
            ]
            params = [svd_solver, random_state, tol, iterated_power]
            params_default = ['arpack', None, 0., 'auto']

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + ". Will run Sklearn PCA.")
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

        from h2o4gpu.decomposition.pca import PCASklearn
        self.model_sklearn = PCASklearn(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state
        )
        self.model_h2o4gpu = PCAH2O(
            n_components=n_components,
            whiten=whiten
        )

        if self.do_sklearn:
            self.model = self.model_sklearn
        else:
            self.model = self.model_h2o4gpu

    def set_attributes(self):
        s = _setter(oself=self, e1=NameError, e2=AttributeError)
        s('oself.components_ = oself.model.components_')
        s('oself.explained_variance_= oself.model.explained_variance_')
        s('oself.explained_variance_ratio_ = '
          'oself.model.explained_variance_ratio_')
        s('oself.singular_values_ = oself.model.singular_values_')
        s('oself.mean_ = oself.model.mean_')
        s('oself.n_components_ = oself.model.n_components_')
        s('oself.noise_variance_ = oself.model.noise_variance_')
