# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
ARIMA model.

:copyright: 2017-2019 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np


class ARIMA(object):
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

    def fit(self, y):
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1
        if self.dtype == np.float32:
            self._lib.arima_fit_float(self.p, self.d, self.q,
                                      y, y.shape[0], self.theta_,
                                      self.phi_)
