# - * - encoding : utf - 8 - * -
"""
:copyright: 2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import numpy as np


class FFMH2O(object):
    """Field-aware factorization machine solver.

    Based on the following white-papers:
    * https://arxiv.org/pdf/1701.04099.pdf
    * https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

    The underlying C/C++/CUDA codebase has been heavily influenced by:
    * https://github.com/guestwalk/libffm
    * https://github.com/RTBHOUSE/cuda-ffm/

    Wrapper class calling an underlying (e.g. GPU or CPU)
     implementation of FFM.

     Parameters
     ----------
     verbose : int, optional, default: 0

     learningRate : float, optional, default: 0.2

     regLambda : int, optional, default: 0.00002

     nIter : int, optional, default: 10
        Maximum number of iterations of the algorithm.

     batchSize : int, optional, default: 1000
        Number of rows which will be used for learning at a time.
        The bigger the batch size the faster the computation but
        also memory consumption.

     k : int, optional, default: 4
        Number of latent features.

     normalize : boolean, optional, default True
        Whether the data should be normalized or not.

     autoStop : boolean, optional, default False
        Whether to stop before nIter if log loss is good enough.

     nGpus : int, optional, default: 1
        Number of threads or GPUs on which the algorithm should run.
        < 0 means all possible GPUs on the machine.
        0 means no GPUs, run on CPU.

    """

    def __init__(
            self,
            verbose=0,
            learning_rate=0.2,
            reg_lambda=0.00002,
            max_iter=10,
            batch_size=1000,
            k=4,
            normalize=True,
            auto_stop=False,
            # TODO change to -1 when multi GPU gets implemented
            n_gpus=1,
            dtype=np.float32
    ):
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.k = k
        self.normalize = normalize
        self.auto_stop = auto_stop
        self.dtype = dtype
        from ..util.gpu import device_count
        (self.nGpus, self.devices) = device_count(n_gpus)

        self.weights = None

        # Hacks for Python/SWIG not to release the objects prematurely
        self.row_arr_holder = []
        self.node_arr_holder = []

    @classmethod
    def _get_param_names(cls):
        # TODO implement
        pass

    def get_params(self, deep=True):
        # TODO implement
        pass

    def set_params(self, **params):
        # TODO implement
        pass

    def fit(self, X, y):
        lib = self._load_lib()

        params = lib.params_ffm()
        params.verbose = self.verbose
        params.learningRate = self.learning_rate
        params.regLambda = self.reg_lambda
        params.nIter = self.max_iter
        params.batchSize = self.batch_size
        params.k = self.k
        params.normalize = self.normalize
        params.autoStop = self.auto_stop
        params.nGpus = self.nGpus

        params.numRows = np.shape(X)[0]

        rows, featureIdx, fieldIdx = self._numpy_to_ffm_rows(X, y, lib)

        weights = np.zeros(params.k * featureIdx * fieldIdx, dtype=self.dtype)

        if self.dtype == np.float32:
            lib.ffm_fit_float(rows, weights, params)
        else:
            lib.ffm_fit_double(rows, weights, params)

        # Cleans up the memory
        self.row_arr_holder = []
        self.node_arr_holder = []

        self.weights = weights
        return self


    def _numpy_to_ffm_rows(self, X, y, lib):
        (node_creator, node_arr_creator, row_creator, row_arr_creator) = \
            (lib.floatNode, lib.NodeFloatArray, lib.floatRow, lib.RowFloatArray) if self.dtype == np.float32 \
                else (lib.doubleNode, lib.NodeDoubleArray, lib.doubleRow, lib.RowDoubleArray)
        nr_rows = np.shape(X)[0]
        row_arr = row_arr_creator(nr_rows)
        self.row_arr_holder.append(row_arr)
        feature_idx = 0
        field_idx = 0
        for r in range(nr_rows):
            nr_nodes = len(X[r])
            node_arr = node_arr_creator(nr_nodes)
            self.node_arr_holder.append(node_arr)
            for n in range(nr_nodes):
                node = node_creator()
                node.fieldIdx = int(X[r][n][0])
                node.featureIdx = int(X[r][n][1])
                node.value = X[r][n][2]
                node_arr.__setitem__(n, node)
                feature_idx = max(feature_idx, node.featureIdx + 1)
                field_idx = max(field_idx, node.fieldIdx + 1)
            # Scale is being set automatically on the C++ side
            row = row_creator(int(y[r]), 1.0, nr_nodes, node_arr)
            row_arr.__setitem__(r, row)
        return row_arr, feature_idx, field_idx

    def predict(self, X):
        # TODO implement
        pass

    def transform(self, X, y=None):
        # TODO implement
        pass

    def fit_transform(self, X, y=None):
        # TODO implement
        pass

    def fit_predict(self, X, y=None):
        # TODO implement
        pass

    # TODO push to a common class
    def _load_lib(self):
        """Load library."""
        from ..libs.lib_utils import GPUlib, CPUlib

        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if (self.nGpus == 0) or (gpu_lib is None) or (self.devices == 0):
            return cpu_lib
        elif (self.nGpus > 0) or (cpu_lib is None) or (self.devices == 0):
            return gpu_lib
        else:
            raise RuntimeError("Failed to initialize FFM solver")
