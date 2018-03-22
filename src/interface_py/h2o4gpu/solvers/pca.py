# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np
from ..libs.lib_pca import parameters
from ..solvers.utils import _setter
from ..solvers.truncated_svd import TruncatedSVDH2O, TruncatedSVD, _as_dptr
from ..utils.extmath import svd_flip


class PCAH2O(TruncatedSVDH2O):
    """Principal Component Analysis (PCA)

    Dimensionality reduction using truncated Singular Value Decomposition
    for GPU

    This implementation uses the Cusolver implementation of the truncated SVD.
    Contrary to SVD, this estimator does center the data before computing
    the singular value decomposition.

    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data

    whiten : bool, optional
        When True (False by default) the `components_` vectors are multiplied
        by the square root of (n_samples) and divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    verbose: bool
        Verbose or not

    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.
    """

    def __init__(self, n_components=2, whiten=False,
                 verbose=0, gpu_id=0):
        super().__init__(n_components)
        self.whiten = whiten
        self.n_components_ = n_components
        self.mean_ = None
        self.noise_variance_ = None
        self.algorithm = "cusolver"
        self.verbose = verbose
        self.gpu_id = gpu_id

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
        Q = np.empty(
            (self.n_components, X.shape[1]), dtype=np.float64, order='F')
        U = np.empty(
            (X.shape[0], self.n_components), dtype=np.float64, order='F')
        w = np.empty(self.n_components, dtype=np.float64)
        mean = np.empty(X.shape[1], dtype=np.float64)
        param = parameters()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.whiten = self.whiten
        param.algorithm = self.algorithm.encode('utf-8')
        param.verbose = 1 if self.verbose else 0
        param.gpu_id = self.gpu_id

        lib = self._load_lib()
        lib.pca(
            _as_dptr(X), _as_dptr(Q), _as_dptr(w), _as_dptr(U),
            _as_dptr(mean), param)

        self._w = w
        self._U, self._Q = svd_flip(U, Q)  # TODO Port to cuda?
        self._X = X
        n = X.shape[0]
        # To match sci-kit #TODO Port to cuda?
        self.explained_variance = self.singular_values_**2 / (n - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio = \
            self.explained_variance / total_var.sum()
        self.mean_ = mean

        # TODO noise_variance_ calculation
        # can be done inside lib.pca if a bottleneck
        n_samples, n_features = X.shape
        total_var = np.var(X, ddof=1, axis=0)
        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = \
                (total_var.sum() - self.explained_variance_.sum())
            self.noise_variance_ /= \
                min(n_features, n_samples) - self.n_components
        else:
            self.noise_variance_ = 0.

        X_transformed = U * w
        return X_transformed

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

    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional
        When True (False by default) the `components_` vectors are multiplied
        by the square root of (n_samples) and divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        'auto' is selected by a default policy based on `X.shape`
        and `n_components`: if the input data is larger than 500x500 and the number
        of components to extract is lower than 80 percent of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards. 'full' runs exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing
        'arpack'runs SVD truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds`.
        It requires strictly 0 < n_components < columns. 'randomized' runs randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    verbose: bool
        Verbose or not

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run. Only used by
        h2o4gpu backend.

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
                 backend='auto',
                 gpu_id=0):
        super().__init__(n_components, random_state, tol, verbose, backend, gpu_id)
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
                              " has been changed from default to " +
                              str(param) + ". Will run Sklearn PCA.")
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
            random_state=random_state)

        self.model_h2o4gpu = PCAH2O(
            n_components=self.n_components,
            whiten=self.whiten,
            verbose=self.verbose,
            gpu_id=self.gpu_id)

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
