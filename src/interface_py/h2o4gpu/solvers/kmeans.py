# -*- encoding: utf-8 -*-
"""
KMeans clustering solver.

:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
from ctypes import c_int, c_float, c_double, c_void_p, pointer, POINTER, cast

import numpy as np

from h2o4gpu.libs.lib_kmeans import GPUlib, CPUlib
from h2o4gpu.solvers.utils import device_count, _to_np, _check_data_content
from h2o4gpu.typecheck.typechecks import assert_is_type, assert_satisfies
from h2o4gpu.types import cptr


class KMeans(object):
    """K-Means clustering

    Wrapper class calling an underlying (e.g. GPU or CPU) implementation of the
    K-Means clustering algorithm.

    :param n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    :param max_iter : int, optional, default: 1000
        Maximum number of iterations of the algorithm.

    :param tol : int, optional, default: 1e-4
        Relative tolerance to declare convergence.

    :param gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

    :param n_gpus : int, optional, default: -1
        Number of GPUs on which the algorithm should run.
        < 0 means all possible GPUs on the machine.
        0 means no GPUs, run on CPU.

    :param init_from_data : boolean, optional, default: False
        If set to True, cluster centers will be initialized
        using random training data points.
        If set to False, cluster centers will be generated
        completely randomly.

    :param init_data : "random", "selectstrat" or
                "randomselect", optional, default: "randomselect"

    :param verbose : int, optional, default 0
        Logger verbosity level.

    :param seed : int or array_like, optional, default: None
        Seed for RandomState. Must be convertible to 32 bit unsigned integers.

    :param do_checks : int, optional, default: 1
        If set to 0 GPU error check will not be performed.

    Attributes:
        cluster_centers_ : array, [n_clusters, n_features], Cluster centers
        labels_ : array, [n_rows,], Labels assigned to each row during fitting.

    Example:
        >>> from h2o4gpu import KMeans
        >>> import numpy as np
        >>> X = np.array([[1, 2], [1, 4], [1, 0],
        ...               [4, 2], [4, 4], [4, 0]])
        >>> kmeans = KMeans(n_clusters=2).fit(X)
        >>> kmeans.labels_
        >>> kmeans.predict(X)
        >>> kmeans.cluster_centers_
    """

    def __init__(self, n_clusters=8,
                 max_iter=1000, tol=1e-4, gpu_id=0, n_gpus=-1,
                 init_from_data=False,
                 init_data="randomselect",
                 verbose=0, seed=None, do_checks=1):

        assert_is_type(n_clusters, int)
        assert_is_type(max_iter, int)
        assert_is_type(tol, float)
        assert_is_type(gpu_id, int)
        assert_is_type(n_gpus, int)
        assert_is_type(init_from_data, bool)
        assert_is_type(init_data, str)
        assert_is_type(verbose, int)
        assert_is_type(seed, int, None)
        assert_is_type(do_checks, int)

        self._n_clusters = n_clusters
        self._gpu_id = gpu_id
        (self.n_gpus, self.devices) = device_count(n_gpus)

        self._max_iter = max_iter
        self.init_from_data = init_from_data
        self.init_data = init_data
        self.tol = tol
        self._did_sklearn_fit = 0
        self.verbose = verbose
        self.do_checks = do_checks

        self.lib = self._load_lib()

        if seed is None:
            import random
            self.seed = random.randint(0, 32000)
        else:
            self.seed = seed

        np.random.seed(seed)

        self.cluster_centers_ = None

        self.labels_ = None

        self.sklearn_model = None

    def get_params(self):
        """Get parameters for this solver as a key-value dictionary.

        :return:
            Mapping of string (parameter name) to its value.
        """
        params = {'n_clusters': self._n_clusters, 'n_gpus': self.n_gpus,
                  'max_iter': self._max_iter, 'init': 'random',
                  'algorithm': 'auto', 'precompute_distances': True,
                  'tol': self.tol, 'n_jobs': -1,
                  'random_state': self.seed, 'verbose': self.verbose,
                  'copy_x': True}
        return params

    def set_params(self, n_clusters=None, n_gpus=None, max_iter=None, tol=None,
                   random_state=None, verbose=None):
        """Set the parameters of this solver.

        :return: self
        """
        if n_clusters is not None:
            self._print_set("n_clusters", self._n_clusters, n_clusters)
            self._n_clusters = n_clusters
        if n_gpus is not None:
            self._print_set("n_gpus", self.n_gpus, n_gpus)
            self.n_gpus = n_gpus
        if max_iter is not None:
            self._print_set("max_iter", self._max_iter, max_iter)
            self._max_iter = max_iter
        if random_state is not None:
            self.seed = random_state
        if verbose is not None:
            self.verbose = verbose
        if tol is not None:
            self.tol = tol

    def fit(self, X, y=None):
        """Compute cluster centers using KMeans algorithm.

        The memory used by this algorithm depends on:
        - m - number of rows in X
        - n - number of dimensions in X
        - k - number of clusters
        - type of data in X (float32 or float64)

        and should be approximately:

        For float32 = 4*(m*n + k*n + 3*m + k + m*k) + 2*(4*m + k)
        For float64 = 8*(m*n + k*n + 3*m + k + m*k) + 2*(4*m + k)

        In case of running on the GPU, a CUDA context size should be
        also taken into account.

        :param X: array-like, shape=(n_samples, n_features)
            Training instances.
        :param y: array-like, optional, shape=(n_samples, 1)
            Initial labels for training.
        """
        Xnp = _to_np(X)

        _check_data_content(self.do_checks, "X", Xnp)
        rows = np.shape(Xnp)[0]

        y_np = self._validate_y(y, rows)

        self._fit(Xnp, y_np)

        self._did_sklearn_fit = 0

        return self

    def sklearn_fit(self, X, y=None):
        """Instantiates a scikit-learn model using previously found,
        with fit(), centroids. """

        assert self.cluster_centers_ is not None, \
            "Centroids are None. Run fit() first."

        if self._did_sklearn_fit == 0:
            X_np = _to_np(X)
            _check_data_content(self.do_checks, "X", X_np)
            rows = np.shape(X_np)[0]

            y_np = self._validate_y(y, rows)

            self._did_sklearn_fit = 1
            import sklearn.cluster as sk_cluster
            self.sklearn_model = sk_cluster.KMeans(self._n_clusters, max_iter=1,
                                                   init=self.cluster_centers_,
                                                   n_init=1)
            self.sklearn_model.fit(X_np, y_np)

    def predict(self, X):
        """ Assign the each record in X to the closest cluster.

        :param X: array-like or sparse matrix of shape [n_samples, n_features]
                  Contains data points to be clustered.
        :return: array of shape [n_samples,]
                A cluster index for each record
        """
        cols, rows = self._validate_centroids(X)

        Xnp = _to_np(X)
        _check_data_content(self.do_checks, "X", Xnp)
        Xnp, c_data, _ = self._to_cdata(Xnp)
        c_init_from_data = 0
        c_init_data = 0

        _, c_centroids, _ = self._to_cdata(self.cluster_centers_, convert=False)
        c_res = c_void_p(0)

        lib = self._load_lib()

        data_ord = ord('c' if np.isfortran(Xnp) else 'r')

        if self.double_precision == 0:
            lib.make_ptr_float_kmeans(1, self.verbose, self.seed, self._gpu_id,
                                      self.n_gpus, rows, cols, c_int(data_ord),
                                      self._n_clusters,
                                      self._max_iter, c_init_from_data,
                                      c_init_data,
                                      self.tol, c_data, None, c_centroids,
                                      None, pointer(c_res))
        else:
            lib.make_ptr_double_kmeans(1, self.verbose, self.seed, self._gpu_id,
                                       self.n_gpus, rows, cols,
                                       c_int(data_ord), self._n_clusters,
                                       self._max_iter, c_init_from_data,
                                       c_init_data,
                                       self.tol, c_data, None, c_centroids,
                                       None, pointer(c_res))

        preds = np.fromiter(cast(c_res, POINTER(c_int)), dtype=np.int32,
                            count=rows)
        preds = np.reshape(preds, rows)
        return preds

    def sklearn_predict(self, X, y=None):
        """
        Instantiates, if necessary, a scikit-learn model using centroids
        found by running fit() and predicts labels using that model.
        This method always runs on CPU, not on GPUs.
        """
        _check_data_content(self.do_checks, "X", X)

        rows = np.shape(X)[0]
        y_np = self._validate_y(y, rows)

        self.sklearn_fit(X, y_np)
        return self.sklearn_model.predict(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        Each dimension is the distance to a cluster center.

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Data to be transformed.

        :return: array, shape [n_samples, k]
            Distances to each cluster for each row.
        """
        cols, rows = self._validate_centroids(X)

        Xnp = _to_np(X)
        Xnp, c_data, c_data_type = self._to_cdata(Xnp)
        _, c_centroids, _ = self._to_cdata(self.cluster_centers_, convert=False)
        c_res = c_void_p(0)

        lib = self._load_lib()

        data_ord = ord('c' if np.isfortran(Xnp) else 'r')

        if self.double_precision == 0:
            lib.kmeans_transform_float(self.verbose,
                                       self._gpu_id, self.n_gpus,
                                       rows, cols, c_int(data_ord),
                                       self._n_clusters, c_data, c_centroids,
                                       pointer(c_res))
        else:
            lib.kmeans_transform_double(self.verbose,
                                        self._gpu_id, self.n_gpus,
                                        rows, cols, c_int(data_ord),
                                        self._n_clusters, c_data, c_centroids,
                                        pointer(c_res))

        transformed = np.fromiter(cast(c_res, POINTER(c_data_type)),
                                  dtype=c_data_type,
                                  count=rows * self._n_clusters)
        transformed = np.reshape(transformed,
                                 (rows, self._n_clusters),
                                 order='F')
        return transformed

    def sklearn_transform(self, X, y=None):
        """
        Instantiates, if necessary, a scikit-learn model using centroids
        found by running fit() and transforms matrix X using that model.
        This method always runs on CPU, not on GPUs.
        """

        _check_data_content(self.do_checks, "X", X)
        self.sklearn_fit(X, y)
        return self.sklearn_model.transform(X)

    def fit_transform(self, X, y=None):
        """Perform fitting and transform X.

        Same as calling fit(X, y).transform(X).

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Data to be transformed.
        :param y: array-like, optional, shape=(n_samples, 1)
            Initial labels for training.

        :return: array, shape [n_samples, k]
            Distances to each cluster for each row.
        """
        return self.fit(X, y).transform(X)

    def fit_predict(self, X, y=None):
        """Perform fitting and prediction on X.

        Same as calling fit(X, y).labels_.

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data to be used for fitting and predictions.
        :param y: array-like, optional, shape=(n_samples, 1)
            Initial labels for training.

        :return: array of shape [n_samples,]
            A cluster index for each record
        """
        return self.fit(X, y).labels_

    def _fit(self, data, labels):
        """Actual method calling the underlying fitting implementation."""
        data_ord = ord('c' if np.isfortran(data) else 'r')

        data, c_data_ptr, data_ctype = self._to_cdata(data)

        c_init_from_data = 0 if self.init_from_data else 1

        if self.init_data == "random":
            c_init_data = 0
        elif self.init_data == "selectstrat":
            c_init_data = 1
        elif self.init_data == "randomselect":
            c_init_data = 2
        else:
            print(
                """
                Unknown init_data "%s", should be
                "random", "selectstrat" or "randomselect".
                """ % self.init_data
            )
            sys.stdout.flush()
            return

        pred_centers = c_void_p(0)
        pred_labels = c_void_p(0)
        c_labels = cptr(labels, dtype=c_int)

        lib = self._load_lib()

        rows = np.shape(data)[0]
        cols = np.shape(data)[1]

        if self.double_precision == 0:
            status = lib.make_ptr_float_kmeans(0, self.verbose, self.seed,
                                               self._gpu_id, self.n_gpus,
                                               rows, cols,
                                               c_int(data_ord),
                                               self._n_clusters,
                                               self._max_iter,
                                               c_init_from_data,
                                               c_init_data,
                                               self.tol, c_data_ptr, c_labels,
                                               None, pointer(pred_centers),
                                               pointer(pred_labels))
        else:
            status = lib.make_ptr_double_kmeans(0, self.verbose, self.seed,
                                                self._gpu_id, self.n_gpus,
                                                rows, cols,
                                                c_int(data_ord),
                                                self._n_clusters,
                                                self._max_iter,
                                                c_init_from_data,
                                                c_init_data,
                                                self.tol, c_data_ptr, c_labels,
                                                None, pointer(pred_centers),
                                                pointer(pred_labels))
        if status:
            raise ValueError('KMeans failed in C++ library.')

        centroids = np.fromiter(cast(pred_centers, POINTER(data_ctype)),
                                dtype=data_ctype,
                                count=self._n_clusters * cols)
        centroids = np.reshape(centroids, (self._n_clusters, cols))

        if np.isnan(centroids).any():
            centroids = centroids[~np.isnan(centroids).any(axis=1)]
            self._print_verbose(0,
                                "Removed %d empty centroids" %
                                (self._n_clusters - centroids.shape[0])
                                )
            self._n_clusters = centroids.shape[0]

        self.cluster_centers_ = centroids

        labels = np.fromiter(cast(pred_labels, POINTER(c_int)),
                             dtype=np.int32, count=rows)
        self.labels_ = np.reshape(labels, rows)

        return self.cluster_centers_, self.labels_

    def _to_cdata(self, data, convert=True):
        """Transform input data into a type which can be passed into C land."""
        if convert and data.dtype != np.float64 and data.dtype != np.float32:
            self._print_verbose(1, "Detected numeric data format which is not "
                                   "supported. Casting to np.float32.")
            data = np.array(data, copy=False, dtype=np.float32)

        if data.dtype == np.float64:
            self._print_verbose(1, "Detected np.float64 data")
            self.double_precision = 1
            my_ctype = c_double
        elif data.dtype == np.float32:
            self._print_verbose(1, "Detected np.float32 data")
            self.double_precision = 0
            my_ctype = c_float
        else:
            raise ValueError(
                "Unsupported data type %s, "
                "should be either np.float32 or np.float64" % data.dtype
            )
        return data, cptr(data, dtype=my_ctype), my_ctype

    def _validate_y(self, y, rows):
        if y is None:
            ynp = np.random.randint(rows, size=rows) % self._n_clusters
        else:
            ynp = _to_np(y)
            _check_data_content(self.do_checks, "y", ynp)

        ynp = ynp.astype(np.int)
        return np.mod(ynp, self._n_clusters)

    def _print_verbose(self, level, msg):
        if self.verbose > level:
            print(msg)
            sys.stdout.flush()

    def _print_set(self, param_name, old_val, new_val):
        self._print_verbose(1, "Changing %s from %d to %d." %
                            (param_name, old_val, new_val))

    def _load_lib(self):
        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if (self.n_gpus == 0) or (gpu_lib is None) or (self.devices == 0):
            raise NotImplementedError("KMeans for CPU not yet supported.")
        elif (self.n_gpus > 0) or (cpu_lib is None) or (self.devices == 0):
            self._print_verbose(0, "\nUsing GPU KMeans solver with %d GPUs.\n" %
                                self.n_gpus)
            return gpu_lib
        else:
            raise RuntimeError("Couldn't instantiate KMeans Solver")

    def _validate_centroids(self, X):
        assert self.cluster_centers_ is not None, \
            "Centroids are None. Run fit() first."
        rows = np.shape(X)[0]
        cols = np.shape(X)[1]
        centroids_dim = np.shape(self.cluster_centers_)[1]
        assert cols == centroids_dim, \
            "The dimension of X [%d] and centroids [%d] is not equal." % \
            (cols, centroids_dim)
        return cols, rows

    # Properties and setters of properties

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value):
        assert_is_type(value, int)
        assert_satisfies(value, value > 0,
                         "Number of clusters must be positive.")
        self._n_clusters = value

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        assert_is_type(value, int)
        assert_satisfies(value, value >= 0,
                         "GPU ID must be non-negative.")
        self._gpu_id = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        assert_is_type(value, int)
        assert_satisfies(value, value > 0,
                         "Number of maximum iterations must be non-negative.")
        self._max_iter = value
