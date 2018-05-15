# -*- encoding: utf-8 -*-
"""
:copyright: 2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import os
import numpy as np
from h2o4gpu.solvers.ffm import FFMH2O

class TestKmeans(object):
    @classmethod
    def setup_class(cls):
        os.environ['SCIKIT_LEARN_DATA'] = "open_data"

    def test_fit_iris(self):
        X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
             [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
             [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],]

        y = [1, 1, 0]

        ffmh2o = FFMH2O(n_gpus=1, max_iter=50, normalize=False, verbose=700)
        ffmh2o.fit(X,y)
        predictions = ffmh2o.predict(X)

        assert np.allclose(predictions, [0.9930521 , 0.9775533 , 0.01786625])