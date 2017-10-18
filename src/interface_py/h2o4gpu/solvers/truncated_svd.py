import ctypes
import numpy as np
from ..libs.lib_tsvd import params

class TruncatedSVD(object):

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asfortranarray(X, dtype=np.float64)
        Q = np.empty((self.n_components, X.shape[1]), dtype=np.float64, order='F')
        U = np.empty((X.shape[0], self.n_components), dtype=np.float64, order='F')
        w = np.empty(self.n_components, dtype=np.float64)
        explained_variance = np.empty(self.n_components, dtype=np.float64);
        explained_variance_ratio = np.empty(self.n_components, dtype=np.float64);
        param = params()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components

        lib = self._load_lib()
        lib.truncated_svd(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U), _as_fptr(explained_variance), _as_fptr(explained_variance_ratio), param)

        self._Q = Q
        self._w = w
        self._U = U
        self._X = X
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

        return self

    def _load_lib(self):
        from ..libs.lib_tsvd import GPUlib

        gpu_lib = GPUlib().get()

        return gpu_lib

    @property
    def components_(self):
        return self._Q

    @property
    def singular_values_(self):
        return self._w

    @property
    def X(self):
        return self._X

    @property
    def U(self):
        return self._U

    @property
    def explained_variance_(self):
        return self.explained_variance

    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ratio

def _as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

