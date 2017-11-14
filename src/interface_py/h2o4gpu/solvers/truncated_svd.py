# - * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import ctypes
import numpy as np
from ..libs.lib_tsvd import parameters
from ..solvers.utils import _setter


class TruncatedSVDH2O(object):
    """Dimensionality reduction using truncated SVD for GPUs

    Perform linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition.

    :param: n_components Desired dimensionality of output data

    """

    def __init__(self, n_components=2):
        self.n_components = n_components

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

        lib = self._load_lib()
        lib.truncated_svd(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U),
                          _as_fptr(explained_variance),
                          _as_fptr(explained_variance_ratio), param)

        self._Q = Q
        self._w = w
        self._U = U
        self._X = X
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

        X_transformed = U * w
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


# Util to send pointers to backend
def _as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class TruncatedSVD(object):
    """
        Truncated SVD Wrapper

        Selects between h2o4gpu.decomposition.TruncatedSVDSklearn
        and h2o4gpu.solvers.truncated_svd.TruncatedSVDH2O

        Documentation:
        import h2o4gpu.decomposition ;
        help(h2o4gpu.decomposition.TruncatedSVDSklearn)
        help(h2o4gpu.solvers.truncated_svd.TruncatedSVD)

    :param: backend : Which backend to use.  Options are 'auto', 'sklearn',
        'h2o4gpu'.  Default is 'auto'.
        Saves as attribute for actual backend used.

    """

    def __init__(self,
                 n_components=2,
                 algorithm="arpack",
                 n_iter=5,
                 random_state=None,
                 tol=0.,
                 verbose=False,
                 backend='auto'):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # Fall back to Sklearn
        # Can remove if fully implement sklearn functionality
        self.do_sklearn = False
        if backend == 'auto':
            params_string = ['algorithm', 'n_iter', 'random_state', 'tol']
            params = [algorithm, n_iter, random_state, tol]
            params_default = ['arpack', 5, None, 0.]

            i = 0
            for param in params:
                if param != params_default[i]:
                    self.do_sklearn = True
                    if verbose:
                        print("WARNING:"
                              " The sklearn parameter " + params_string[i] +
                              " has been changed from default to " + str(param)
                              + ". Will run Sklearn TruncatedSVD.")
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

        from h2o4gpu.decomposition.truncated_svd import TruncatedSVDSklearn
        self.model_sklearn = TruncatedSVDSklearn(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol)
        self.model_h2o4gpu = TruncatedSVDH2O(
            n_components=n_components)

        if self.do_sklearn:
            self.model = self.model_sklearn
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
