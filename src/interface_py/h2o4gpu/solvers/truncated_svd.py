# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import print_function
import ctypes
import sys
import time
import numpy as np
from ..libs.lib_tsvd import parameters as parameters_svd
from ..solvers.utils import _setter
from ..utils.extmath import svd_flip
from ..types import cptr


class TruncatedSVDH2O(object):
    """Dimensionality reduction using truncated SVD for GPUs

    Perform linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition.

    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data

    algorithm: string, Default="power"
        SVD solver to use.
        Either "cusolver" (similar to ARPACK)
        or "power" for the power method.

    n_iter: int, Default=100
        number of iterations (only relevant for power method)
        Should be at most 2147483647 due to INT_MAX in C++ backend.

    int random_state: seed (None for auto-generated)

    float tol: float, Default=1E-5
        Tolerance for "power" method. Ignored by "cusolver".
        Should be > 0.0 to ensure convergence.
        Should be 0.0 to effectively ignore
        and only base convergence upon n_iter

    verbose: bool
        Verbose or not

    n_gpus : int, optional, default: 1
        How many gpus to use.  If 0, use CPU backup method.
        Currently SVD only uses 1 GPU, so >1 has no effect compared to 1.

    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

    """

    def __init__(self, n_components=2, algorithm="power",
                 n_iter=100, random_state=None, tol=1e-5,
                 verbose=0, n_gpus=1, gpu_id=0):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.randint(0, 2 ** 32 - 1)
        self.tol = tol
        self.verbose = verbose
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """Fit Truncated SVD on matrix X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :param y Ignored

        :returns self : object

        """
        self.fit_transform(X)
        return self

    # pylint: disable=unused-argument
    def fit_transform(self, X, y=None):
        """Fit Truncated SVD on matrix X and perform dimensionality reduction
           on X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :param: y Ignored

        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always be a
                         dense array.

        """
        import scipy
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            X = scipy.sparse.csr_matrix.todense(X)

        X = self._check_double(X)
        if self.double_precision == 1:
            X = np.asfortranarray(X, dtype=np.float64)
        else:
            X = np.asfortranarray(X, dtype=np.float32)

        if self.double_precision == 1:
            print("Detected Double")
            Q = np.empty(
                (self.n_components, X.shape[1]), dtype=np.float64, order='F')
            U = np.empty(
                (X.shape[0], self.n_components), dtype=np.float64, order='F')
            w = np.empty(self.n_components, dtype=np.float64)

        else:
            print("Detected Float")
            Q = np.empty(
                (self.n_components, X.shape[1]), dtype=np.float32, order='F')
            U = np.empty(
                (X.shape[0], self.n_components), dtype=np.float32, order='F')
            w = np.empty(self.n_components, dtype=np.float32)

        param = parameters_svd()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.algorithm = self.algorithm.encode('utf-8')
        param.tol = self.tol
        param.n_iter = self.n_iter
        param.random_state = self.random_state
        param.verbose = 1 if self.verbose else 0
        param.gpu_id = self.gpu_id

        if param.tol < 0.0:
            raise ValueError("The `tol` parameter must be >= 0.0 "
                             "but got " + str(param.tol))
        if param.n_iter < 1:
            raise ValueError("The `n_iter` parameter must be > 1 "
                             "but got " + str(param.n_iter))
        if param.n_iter > 2147483647:
            raise ValueError("The `n_iter parameter cannot exceed "
                             "the value for "
                             "C++ INT_MAX (2147483647) "
                             "but got`" + str(self.n_iter))

        lib = self._load_lib()
        if self.double_precision == 1:
            lib.truncated_svd_double(
                cptr(X, ctypes.c_double), cptr(Q, ctypes.c_double),
                cptr(w, ctypes.c_double), cptr(U, ctypes.c_double),
                param)
        else:
            lib.truncated_svd_float(
                cptr(X, ctypes.c_float), cptr(Q, ctypes.c_float),
                cptr(w, ctypes.c_float), cptr(U, ctypes.c_float),
                param)

        self._w = w
        self._X = X
        self._U, self._Q = svd_flip(U, Q)
        X_transformed = self._U * self._w
        if self.verbose:
            start_ev = time.time()
        self.explained_variance = \
            np.var(X_transformed, axis=0)
        if self.verbose:
            print("Time taken for explained variance : "
                  + str(time.time()-start_ev))
        if self.verbose:
            start_var = time.time()
        full_var = \
            np.var(X, axis=0).sum()
        if self.verbose:
            print("Time taken for full variance : " + str(time.time() - start_var))
        if self.verbose:
            start_evr = time.time()
        self.explained_variance_ratio = \
            self.explained_variance / full_var
        if self.verbose:
            print("Time taken for explained variance ratio : " + str(time.time() - start_evr))
        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always
                         be a dense array.

        """
        fit = self.fit(X)
        X_new = fit.U * fit.singular_values_
        return X_new

    def inverse_transform(self, X):
        """Transform X back to its original space.

        :param: X array-like, shape (n_samples, n_components)

        :returns X_original : array, shape (n_samples, n_features)
                              Note that this is always a dense array.

        """
        return np.dot(X, self.components_)

    def _check_double(self, data, convert=True):
        """Transform input data into a type which can be passed into C land."""
        if convert and data.dtype != np.float64 and data.dtype != np.float32:
            self._print_verbose(0, "Detected numeric data format which is not "
                                   "supported. Casting to np.float32.")
            data = np.asfortranarray(data, dtype=np.floa32)

        if data.dtype == np.float64:
            self._print_verbose(0, "Detected np.float64 data")
            self.double_precision = 1
            data = np.asfortranarray(data, dtype=np.float64)
        elif data.dtype == np.float32:
            self._print_verbose(0, "Detected np.float32 data")
            self.double_precision = 0
            data = np.asfortranarray(data, dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported data type %s, "
                "should be either np.float32 or np.float64" % data.dtype)
        return data

    def _print_verbose(self, level, msg):
        if self.verbose > level:
            print(msg)
            sys.stdout.flush()

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        from h2o4gpu.utils.fixes import signature
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])


    def get_params(self, deep=True):
        """Get parameters for this estimator.

        :param bool deep : If True, will return the parameters for this
            estimator and contained subobjects that are estimators.

        :returns dict params : Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils / __init__.py but it gets overwritten
            # when running under python3 somehow.
            import warnings

            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if w and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

                # XXX : should we rather test if instance of estimator ?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this solver.

        :return: self
        """
        if not params:
            # Simple optimization to gain speed(inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        from ..externals import six
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    @property
    def components_(self):
        return self._Q

    @property
    def explained_variance_(self):
        return self.explained_variance

    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ratio

    @property
    def singular_values_(self):
        return self._w

    @property
    def U(self):
        return self._U

    # Util to load gpu lib
    def _load_lib(self):
        from ..libs.lib_tsvd import GPUlib

        gpu_lib = GPUlib().get()

        return gpu_lib

def _as_dptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class TruncatedSVD(object):
    """
    Truncated SVD Wrapper

    Selects between h2o4gpu.decomposition.TruncatedSVDSklearn
    and h2o4gpu.solvers.truncated_svd.TruncatedSVDH2O

    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data

    algorithm: string, Default="power"
        SVD solver to use.
        H2O4GPU options:
            Either "cusolver" (similar to ARPACK)
            or "power" for the power method.
        SKlearn options:
            Either "arpack" for the ARPACK wrapper in SciPy
            (scipy.sparse.linalg.svds), or "randomized" for the randomized
            algorithm due to Halko (2009).

    n_iter: int, Default=100
        number of iterations (only relevant for power method)
        Should be at most 2147483647 due to INT_MAX in C++ backend.

    random_state: int, Default=None
        seed (None for auto-generated)

    tol: float, Default=1E-5
        Tolerance for "power" method. Ignored by "cusolver".
        Should be > 0.0 to ensure convergence.
        Should be 0.0 to effectively ignore
        and only base convergence upon n_iter

    verbose: bool
        Verbose or not

    backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    n_gpus : int, optional, default: 1
        How many gpus to use.  If 0, use CPU backup method.
        Currently SVD only uses 1 GPU, so >1 has no effect compared to 1.

    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

    """

    def __init__(self,
                 n_components=2,
                 algorithm="power",
                 n_iter=100,
                 random_state=None,
                 tol=1E-5,
                 verbose=False,
                 backend='auto',
                 n_gpus=1,
                 gpu_id=0):
        if isinstance(algorithm, list):
            self.algorithm = algorithm[0]
        else:
            self.algorithm = algorithm
        self.n_components = n_components
        if isinstance(n_iter, list):
            self.n_iter = n_iter[0]
        else:
            self.n_iter = n_iter
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.randint(0, 2 ** 32 - 1)
        if isinstance(tol, list):
            self.tol = tol[0]
        else:
            self.tol = tol
        self.verbose = 1 if verbose else 0
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        self.do_daal = False

        sklearn_algorithm = "arpack"  # Default scikit
        sklearn_n_iter = 5
        sklearn_tol = 1E-5

        if n_gpus == 0:
            # we don't have CPU back-end for SVD yet.
            backend = 'sklearn'
        else:
            backend = 'h2o4gpu'

        if backend in ['auto', 'sklearn']:
            self.do_sklearn = True
            self.backend = 'sklearn'
            params_string = ['algorithm']
            params = [self.algorithm]
            params_gpu = [['cusolver', 'power']]

            i = 0
            for param in params:
                if param not in params_gpu[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The parameter " + params_string[i]
                              + "is "
                              + str(param)
                              + " and not supported by GPU."
                              + "Will run Sklearn TruncatedSVD.")
                    self.do_sklearn = True
                i = i + 1

            if isinstance(algorithm, list):
                sklearn_algorithm = algorithm[1]
            if isinstance(n_iter, list):
                sklearn_n_iter = n_iter[1]
            if isinstance(tol, list):
                sklearn_tol = tol[1]

        elif backend == 'h2o4gpu':
            self.backend = 'h2o4gpu'

        elif backend == 'daal':
            from h2o4gpu import DAAL_SUPPORTED
            if DAAL_SUPPORTED:
                from h2o4gpu.solvers.daal_solver.svd import SVD
                self.do_daal = True
                self.backend = 'daal'

                self.model_daal = SVD(n_components=self.n_components,
                                      verbose=self.verbose)
            else:
                import platform
                print("WARNING:"
                      "DAAL is supported only for x86_64, "
                      "architecture detected {}. Sklearn model"
                      "used instead".format(platform.architecture()))
                self.do_sklearn = True
                self.backend = 'sklearn'

        from h2o4gpu.decomposition.truncated_svd import TruncatedSVDSklearn
        self.model_sklearn = TruncatedSVDSklearn(
            n_components=self.n_components,
            algorithm=sklearn_algorithm,
            n_iter=sklearn_n_iter,
            random_state=self.random_state,
            tol=sklearn_tol)

        self.model_h2o4gpu = TruncatedSVDH2O(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
            gpu_id=self.gpu_id)

        # select final model type
        if self.do_sklearn:
            self.model = self.model_sklearn
        elif self.do_daal:
            self.model = self.model_daal
        else:
            self.model = self.model_h2o4gpu

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        res = self.model.fit(X)
        self.set_attributes()
        return res

    # pylint: disable=unused-argument
    def fit_transform(self, X, y=None):
        res = self.model.fit_transform(X)
        self.set_attributes()
        return res

    def get_params(self, deep=True):
        res = self.model.get_params(deep)
        self.set_attributes()
        return res

    def set_params(self, **params):
        res = self.model.set_params(**params)
        self.set_attributes()
        return res

    def transform(self, X):
        res = self.model.transform(X)
        self.set_attributes()
        return res

    def inverse_transform(self, X):
        res = self.model.inverse_transform(X)
        self.set_attributes()
        return res

    def set_attributes(self):
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.components_ = oself.model.components_')
        s('oself.explained_variance_= oself.model.explained_variance_')
        s('oself.explained_variance_ratio_ = '
          'oself.model.explained_variance_ratio_')
        s('oself.singular_values_ = oself.model.singular_values_')
