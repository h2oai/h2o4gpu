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
            seed=None,
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
        self.seed = seed
        from ..util.gpu import device_count
        (self.nGpus, self.devices) = device_count(n_gpus)

        self.weights = None

        self.learned_params = None

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

        params.seed = 0 if self.seed is None else self.seed

        params.numRows = np.shape(X)[0]

        features, fields, values, scales, positions = self._numpy_to_ffm_rows(params, X)

        weights = np.zeros(params.k * params.numFeatures * params.numFields, dtype=self.dtype)

        y_np = self._sanatize_labels(y)

        if self.dtype == np.float32:
            lib.ffm_fit_float(features, fields, values, y_np, scales, positions, weights, params)
        else:
            lib.ffm_fit_double(features, fields, values, y_np, scales, positions, weights, params)

        self.learned_params = params
        self.weights = weights
        return self

    def _sanatize_labels(self, y):
        return np.array(list(map(lambda e: 1 if e > 0 else -1, y)), dtype=np.int32)

    def _numpy_to_ffm_rows(self, params, X, update_param_features=True):
        nr_rows = np.shape(X)[0]

        num_nodes = 0
        for r in range(nr_rows):
            num_nodes = num_nodes + len(X[r])
        params.numNodes = num_nodes

        features = np.zeros(num_nodes, dtype=np.int32)
        fields = np.zeros(num_nodes, dtype=np.int32)
        values = np.zeros(num_nodes, dtype=self.dtype)
        scales = np.zeros(nr_rows, dtype=self.dtype)
        positions = np.zeros(nr_rows + 1, dtype=np.int32)

        feature_idx = 0
        field_idx = 0

        curr_idx = 0
        for r in range(nr_rows):
            nr_nodes = len(X[r])
            scale = 0.0
            positions[r + 1] = positions[r] + nr_nodes
            for n in range(nr_nodes):
                field = int(X[r][n][0])
                fields[curr_idx] = field
                field_idx = max(field_idx, field + 1)

                feature = int(X[r][n][1])
                features[curr_idx] = feature
                feature_idx = max(feature_idx, feature + 1)

                value = X[r][n][2]
                values[curr_idx] = value
                scale = scale + (value * value)
                curr_idx = curr_idx + 1
            scales[r] = 1.0 / scale

        if update_param_features:
            params.numFeatures = feature_idx
            params.numFields = field_idx

        return features, fields, values, scales, positions

    def predict(self, X):
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
        params.numFields = self.learned_params.numFields
        params.numFeatures = self.learned_params.numFeatures

        params.numRows = np.shape(X)[0]

        features, fields, values, scales, positions = self._numpy_to_ffm_rows(params, X)

        predictions = np.zeros(params.numRows, dtype=self.dtype)

        if self.dtype == np.float32:
            lib.ffm_predict_float(features, fields, values, scales, positions, predictions, self.weights, params)
        else:
            lib.ffm_predict_double(features, fields, values, scales, positions, predictions, self.weights, params)

        return predictions

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
