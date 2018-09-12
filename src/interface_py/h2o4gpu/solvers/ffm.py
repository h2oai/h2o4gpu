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
            batch_size=-1,
            k=4,
            normalize=True,
            auto_stop=True,
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
        self.actual_iterations = 0

    @classmethod
    def _get_param_names(cls):
        """
        Not yet currently implemented, for API consistency purpose here.
        """
        pass

    def get_params(self, deep=True):
        """
        Not yet currently implemented, for API consistency purpose here.
        """
        pass

    def set_params(self, **params):
        """
        Not yet currently implemented, for API consistency purpose here.
        """
        pass

    def fit(self, X, y, X_validate=None, y_validate=None):
        """
        Fit an FFM model. Validation dataset is not required but
        highly recommended as FFMs tend to overfit.

        :param X: 2D array-like of 3-tuples (field:feature:value)
        :param y:
            array-like containing training labels.
            >0 labels are treated as positive <=0 as negatives
        :param X_validate: 2D array-like of 3-tuples (field:feature:value)
        :param y_validate:
            array-like containing training labels.
            >0 labels are treated as positive <=0 as negatives
        :return:
        """
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

        fields, features, values, positions, num_nodes = \
            self._numpy_to_ffm_rows(params, X)
        params.numNodes = num_nodes

        y_np = self._sanatize_labels(y)

        fields_validation, features_validation, values_validation, \
        positions_validation = None, None, None, None
        if X_validate is not None and y_validate is not None:
            fields_validation, features_validation, values_validation, \
            positions_validation, num_nodes_validate = \
                self._numpy_to_ffm_rows(params, X_validate)
            params.numRowsVal = np.shape(X_validate)[0]
            params.numNodesVal = num_nodes_validate

        y_validation_np = self._sanatize_labels(y_validate)

        self.weights = \
            np.zeros(params.k * (np.max(features) + 1) * (np.max(fields) + 1),
                     dtype=self.dtype)

        if self.dtype == np.float32:
            self.actual_iterations = \
                lib.ffm_fit_float(features, fields, values, y_np, positions,
                                  features_validation, fields_validation,
                                  values_validation, y_validation_np,
                                  positions_validation, self.weights, params)
        else:
            self.actual_iterations = \
                lib.ffm_fit_double(features, fields, values, y_np, positions,
                                   features_validation, fields_validation,
                                   values_validation, y_validation_np,
                                   positions_validation, self.weights, params)

        self.learned_params = params
        return self

    def _sanatize_labels(self, y):
        if y is None:
            return None
        return np.array(
            list(map(lambda e: 1 if e > 0 else -1, y)), dtype=np.int32)

    def _numpy_to_ffm_rows(self, params, X):
        """
        Breaks down a 2D array-like object of 3-tuple into structures which
        can be passed to the C ffm backend: 1D arrays of fields, features,
        values, row positions.

        :param params:
        :param X:
        :return:
        """
        import time
        start = time.time()
        nr_rows = np.shape(X)[0]

        num_nodes = 0
        for _, row in enumerate(X):
            num_nodes = num_nodes + len(row)

        features = np.zeros(num_nodes, dtype=np.int32)
        fields = np.zeros(num_nodes, dtype=np.int32)
        values = np.zeros(num_nodes, dtype=self.dtype)
        positions = np.zeros(nr_rows + 1, dtype=np.int32)

        curr_idx = 0
        for i, row in enumerate(X):
            nr_nodes = len(row)
            positions[i + 1] = positions[i] + nr_nodes
            for _, (field, feature, value) in enumerate(row):
                fields[curr_idx] = field
                features[curr_idx] = feature
                values[curr_idx] = value
                curr_idx = curr_idx + 1

        if params.verbose > 0:
            print("Preparing data for FFM took %d." % (time.time() - start))

        return fields, features, values, positions, num_nodes

    def predict(self, X):
        """
        Returns a prediction per row. Requires `fit()` to be ran beforehand.

        :param X: 2D array-like of 3-tuples (field:feature:value)
        :return: array of predictions, one per X row
        """
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

        fields, features, values, positions, num_nodes = \
            self._numpy_to_ffm_rows(params, X)
        params.numNodes = num_nodes

        predictions = np.zeros(params.numRows, dtype=self.dtype)

        if self.dtype == np.float32:
            lib.ffm_predict_float(
                features, fields, values,
                positions, predictions, self.weights, params)
        else:
            lib.ffm_predict_double(
                features, fields, values,
                positions, predictions, self.weights, params)

        return predictions

    def transform(self, X, y=None):
        """
        Not yet currently implemented, for API consistency purpose here.
        :param X:
        :param y:
        :return:
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Not yet currently implemented, for API consistency purpose here.
        :param X:
        :param y:
        :return:
        """
        pass

    def fit_predict(self, X, y=None):
        """
        Not yet currently implemented, for API consistency purpose here.
        :param X:
        :param y:
        :return:
        """
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
