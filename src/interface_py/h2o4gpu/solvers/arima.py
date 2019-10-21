# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
Autoregressive Integrated Moving Average (ARIMA) model.

:copyright: 2017-2019 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np


class ARIMA(object):
    """ Autoregressive integrated moving average

    Parameters
    ----------
    p : int
        AR size
    d : int
        differencing order
    q : int
        MA size
    double_precision : bool, optional
        use double precision, by default False

    Attributes
    ----------
    phi_ {array-like} shape (p,)
        AR coefficients
    theta_ {array-like} shape (q,)
        MA coefficients
    """

    def __init__(self, p, d, q, double_precision=False):
        self.p = p
        self.d = d
        self.q = q
        self.dtype = np.float64 if double_precision else np.float32
        self._lib = self._load_lib()
        self.phi_ = np.empty(p, dtype=self.dtype)
        self.theta_ = np.empty(q, dtype=self.dtype)

    def _load_lib(self):
        from ..libs.lib_utils import GPUlib

        gpu_lib = GPUlib().get(1)
        return gpu_lib

    def fit(self, y, maxiter=20):
        """Fit ARIMA model

        Parameters
        ----------
        y : array-like
            data to fit into ARIMA model
        maxiter : int, optional
            number of iterations, by default 20
        """
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        if self.dtype == np.float32:
            self._lib.arima_fit_float(self.p, self.d, self.q,
                                      np.flipud(y.astype(self.dtype)),
                                      y.shape[0], self.theta_,
                                      self.phi_, maxiter)
        else:
            self._lib.arima_fit_double(self.p, self.d, self.q,
                                       np.flipud(y.astype(self.dtype)),
                                       y.shape[0], self.theta_,
                                       self.phi_, maxiter)
