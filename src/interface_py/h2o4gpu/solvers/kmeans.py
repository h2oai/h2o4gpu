# - * - encoding : utf - 8 - * -
# pylint: disable=fixme, line-too-long
"""
KMeans clustering solver.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
from ctypes import c_int, c_float, c_double, c_void_p, pointer, \
    POINTER, cast

import numpy as np

from ..solvers.utils import _check_data_content, \
    _get_data, _setter
from ..util.gpu import device_count
from ..typecheck.typechecks import assert_satisfies
from ..types import cptr


class KMeansH2O(object):
    """K-Means clustering

    Wrapper class calling an underlying (e.g. GPU or CPU)
     implementation of the K-Means clustering algorithm.

    Approximate GPU Memory Use:
     n_clusters*rows + rows*cols + cols*n_clusters

     Parameters
     ----------
     n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

     init : string, {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'random':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        *Not supported yet* - if chosen we will use SKLearn's methods.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        *Not supported yet* - if chosen we will use SKLearn's methods.

     n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
        *Not supported yet* - always runs 1.

     max_iter : int, optional, default: 1000
        Maximum number of iterations of the algorithm.

     tol : int, optional, default: 1e-4
        Relative tolerance to declare convergence.

     precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
        *Not supported yet* - always uses auto if running h2o4gpu version.

     verbose : int, optional, default 0
        Logger verbosity level.

     random_state : int or array_like, optional, default: None
        random_state for RandomState.
        Must be convertible to 32 bit unsigned integers.

     copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
        *Not supported yet* - always uses True if running h2o4gpu version.

     n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
        *Not supported yet* - CPU backend not yet implemented.

     algorithm : string, "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
        *Not supported yet* - always uses full if running h2o4gpu version.

     gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

     n_gpus : int, optional, default: -1
        Number of GPUs on which the algorithm should run.
        < 0 means all possible GPUs on the machine.
        0 means no GPUs, run on CPU.

     do_checks : int, optional, default: 1
        If set to 0 GPU error check will not be performed.

    Attributes:
        cluster_centers_ : array, [n_clusters, n_features], Cluster centers
        labels_ : array, [n_rows,],
            Labels assigned to each row during fitting.
        inertia_ : float Sum of distances of samples
            to their closest cluster center.

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

    # pylint: disable=unused-argument
    def __init__(
            self,
            # sklearn API (but with possibly different choices for defaults)
            n_clusters=8,
            init='k-means++',
            n_init=1,
            max_iter=300,
            tol=1e-4,
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=1,
            algorithm='auto',
            # Beyond sklearn (with optimal defaults)
            gpu_id=0,
            n_gpus=-1,
            do_checks=1):

        # fix-up tol in case input was numpy
        example = np.fabs(1.0)
        # pylint: disable=unidiomatic-typecheck
        if type(tol) == type(example):
            tol = tol.item()

        if isinstance(init, np.ndarray):
            assert ValueError("Passing initial centroids not yet supported.")

        if isinstance(init, str) and init not in ['random', 'k-means++']:
            assert ValueError(
                "Invalid initialization method. "
                "Should be 'k-means++' or 'random' but got '%s'." % init)

        self.init = init
        self._n_clusters = n_clusters
        self._gpu_id = gpu_id
        (self.n_gpus, self.devices) = device_count(n_gpus)

        self._max_iter = max_iter
        self.tol = tol
        self._did_sklearn_fit = 0
        self.verbose = verbose
        self.do_checks = do_checks

        if random_state is None:
            import random
            self.random_state = random.randint(0, 32000)
        else:
            self.random_state = random_state

        self.cluster_centers_ = None

        self.labels_ = None

        self.inertia_ = None  # TODO: Not set yet

        self.sklearn_model = None

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
        from ..utils.fixes import signature
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != 'self' and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("h2o4gpu GLM estimator should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention." %
                                   (cls, init_signature))
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
        """
        X_np, _, _, _, _, _ = _get_data(X, ismatrix=True)

        _check_data_content(self.do_checks, "X", X_np)

        self._fit(X_np)

        self._did_sklearn_fit = 0

        if y is not None:
            pass  # not using labels

        return self

    # y is here just for compatibility with sklearn api
    # pylint: disable=unused-argument
    def sklearn_fit(self, X, y=None):
        """Instantiates a scikit-learn model using previously found,
        with fit(), centroids. """

        assert self.cluster_centers_ is not None, \
            "Centroids are None. Run fit() first."

        if self._did_sklearn_fit == 0:
            X_np, _, _, _, _, _ = _get_data(X, ismatrix=True)
            _check_data_content(self.do_checks, "X", X_np)

            self._did_sklearn_fit = 1
            import sklearn.cluster as sk_cluster
            self.sklearn_model = sk_cluster.KMeans(
                self._n_clusters,
                max_iter=1,
                init=self.cluster_centers_,
                n_init=1)
            self.sklearn_model.fit(X_np)
            # The code above initializes the SKlearn KMeans model,
            # but due to validations we need to run 1 extra iteration,
            # which might alter the cluster centers so we override them
            self.sklearn_model.cluster_centers_ = self.cluster_centers_

    def predict(self, X):
        """ Assign the each record in X to the closest cluster.

        :param X: array-like or sparse matrix of shape [n_samples, n_features]
                  Contains data points to be clustered.
        :return: array of shape [n_samples,]
                A cluster index for each record
        """
        cols, rows = self._validate_centroids(X)

        X_np, _, _, _, _, _ = _get_data(X, ismatrix=True)
        _check_data_content(self.do_checks, "X", X_np)
        X_np, c_data, _ = self._to_cdata(X_np)
        c_init = 0

        _, c_centroids, _ = self._to_cdata(self.cluster_centers_, convert=False)
        c_res = c_void_p(0)

        lib = self._load_lib()

        data_ord = ord('c' if np.isfortran(X_np) else 'r')

        if self.double_precision == 0:
            c_kmeans = lib.make_ptr_float_kmeans
        else:
            c_kmeans = lib.make_ptr_double_kmeans

        c_kmeans(1, self.verbose,
                 self.random_state, self._gpu_id, self.n_gpus, rows, cols,
                 c_int(data_ord), self._n_clusters, self._max_iter, c_init,
                 self.tol, c_data, c_centroids, None, pointer(c_res))

        preds = np.fromiter(
            cast(c_res, POINTER(c_int)), dtype=np.int32, count=rows)
        preds = np.reshape(preds, rows)
        return preds

    # y is here just for compatibility with sklearn api
    # pylint: disable=unused-argument
    def sklearn_predict(self, X, y=None):
        """
        Instantiates, if necessary, a scikit-learn model using centroids
        found by running fit() and predicts labels using that model.
        This method always runs on CPU, not on GPUs.
        """
        _check_data_content(self.do_checks, "X", X)

        self.sklearn_fit(X)
        return self.sklearn_model.predict(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.

        Each dimension is the distance to a cluster center.

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
                Data to be transformed.

        :return: array, shape [n_samples, k]
            Distances to each cluster for each row.
        """
        cols, rows = self._validate_centroids(X)

        X_np, _, _, _, _, _ = _get_data(X, ismatrix=True)
        X_np, c_data, c_data_type = self._to_cdata(X_np)
        _, c_centroids, _ = self._to_cdata(self.cluster_centers_, convert=False)
        c_res = c_void_p(0)

        lib = self._load_lib()

        data_ord = ord('c' if np.isfortran(X_np) else 'r')

        if self.double_precision == 0:
            lib.kmeans_transform_float(
                self.verbose, self._gpu_id, self.n_gpus, rows, cols,
                c_int(data_ord), self._n_clusters, c_data, c_centroids,
                pointer(c_res))
        else:
            lib.kmeans_transform_double(
                self.verbose, self._gpu_id, self.n_gpus, rows, cols,
                c_int(data_ord), self._n_clusters, c_data, c_centroids,
                pointer(c_res))

        transformed = np.fromiter(
            cast(c_res, POINTER(c_data_type)),
            dtype=c_data_type,
            count=rows * self._n_clusters)
        # TODO don 't set order if X is ' F'
        transformed = np.reshape(
            transformed, (rows, self._n_clusters), order='F')
        return transformed

    def sklearn_transform(self, X, y=None):
        """
        Instantiates, if necessary, a scikit-learn model using centroids
        found by running fit() and transforms matrix X using that model.
        This method always runs on CPU, not on GPUs.
        """

        _check_data_content(self.do_checks, "X", X)
        self.sklearn_fit(X)
        # pylint: disable=too-many-function-args
        return self.sklearn_model.transform(X, y)

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

    def _fit(self, data):
        """Actual method calling the underlying fitting implementation."""
        data_ord = ord('c' if np.isfortran(data) else 'r')

        data, c_data_ptr, data_ctype = self._to_cdata(data)

        if self.init == "k-means++":
            c_init = 1
        else:
            c_init = 0

        pred_centers = c_void_p(0)
        pred_labels = c_void_p(0)

        lib = self._load_lib()

        rows = np.shape(data)[0]
        cols = np.shape(data)[1]

        if self.double_precision == 0:
            status = lib.make_ptr_float_kmeans(
                0, self.verbose,
                self.random_state, self._gpu_id, self.n_gpus, rows, cols,
                c_int(data_ord), self._n_clusters, self._max_iter, c_init,
                self.tol, c_data_ptr, None, pointer(pred_centers),
                pointer(pred_labels))
        else:
            status = lib.make_ptr_double_kmeans(
                0, self.verbose,
                self.random_state, self._gpu_id, self.n_gpus, rows, cols,
                c_int(data_ord), self._n_clusters, self._max_iter, c_init,
                self.tol, c_data_ptr, None, pointer(pred_centers),
                pointer(pred_labels))
        if status:
            raise ValueError('KMeans failed in C++ library.')

        centroids = np.fromiter(
            cast(pred_centers, POINTER(data_ctype)),
            dtype=data_ctype,
            count=self._n_clusters * cols)
        centroids = np.reshape(centroids, (self._n_clusters, cols))

        if np.isnan(centroids).any():
            centroids = centroids[~np.isnan(centroids).any(axis=1)]
            self._print_verbose(0, "Removed %d empty centroids" %
                                (self._n_clusters - centroids.shape[0]))
            self._n_clusters = centroids.shape[0]

        self.cluster_centers_ = centroids

        labels = np.ctypeslib.as_array(
            cast(pred_labels, POINTER(c_int)), (rows,))
        self.labels_ = np.reshape(labels, rows)

        return self.cluster_centers_, self.labels_

    # FIXME : This function duplicates others
    # in solvers / utils.py as used in GLM

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
                "should be either np.float32 or np.float64" % data.dtype)
        return data, cptr(data, dtype=my_ctype), my_ctype

    def _print_verbose(self, level, msg):
        if self.verbose > level:
            print(msg)
            sys.stdout.flush()

    def _print_set(self, param_name, old_val, new_val):
        self._print_verbose(1, "Changing %s from %d to %d." %
                            (param_name, old_val, new_val))

    def _load_lib(self):
        """Load library."""
        from ..libs.lib_kmeans import GPUlib, CPUlib

        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if (self.n_gpus == 0) or (gpu_lib is None) or (self.devices == 0):
            self._print_verbose(0, "H2O KMeans for CPU not yet supported.")
            return None
        elif (self.n_gpus > 0) or (cpu_lib is None) or (self.devices == 0):
            self._print_verbose(
                0, "\nUsing GPU KMeans solver with %d GPUs.\n" % self.n_gpus)
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
        assert_satisfies(value, value > 0,
                         "Number of clusters must be positive.")
        self._n_clusters = value

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        assert_satisfies(value, value >= 0, "GPU ID must be non-negative.")
        self._gpu_id = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        assert_satisfies(value, value > 0,
                         "Number of maximum iterations must be non-negative.")
        self._max_iter = value


class KMeans(object):
    """
     K-Means clustering Wrapper

     Selects between h2o4gpu.cluster.k_means_.KMeansSklearn
     and h2o4gpu.solvers.kmeans.KMeansH2O

     Parameters
     ----------
     n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

     init : string, {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'random':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        *Not supported yet* - if chosen we will use SKLearn's methods.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        *Not supported yet* - if chosen we will use SKLearn's methods.

     n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
        *Not supported yet* - always runs 1.

     max_iter : int, optional, default: 1000
        Maximum number of iterations of the algorithm.

     tol : int, optional, default: 1e-4
        Relative tolerance to declare convergence.

     precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
        *Not supported yet* - always uses auto if running h2o4gpu version.

     verbose : int, optional, default 0
        Logger verbosity level.

     random_state : int or array_like, optional, default: None
        random_state for RandomState.
        Must be convertible to 32 bit unsigned integers.

     copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
        *Not supported yet* - always uses True if running h2o4gpu version.

     n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
        *Not supported yet* - CPU backend not yet implemented.

     algorithm : string, "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
        *Not supported yet* - always uses full if running h2o4gpu version.

     gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

     n_gpus : int, optional, default: -1
        Number of GPUs on which the algorithm should run.
        < 0 means all possible GPUs on the machine.
        0 means no GPUs, run on CPU.

     do_checks : int, optional, default: 1
        If set to 0 GPU error check will not be performed.

     backend : string, (Default="auto")
        Which backend to use.
        Options are 'auto', 'sklearn', 'h2o4gpu'.
        Saves as attribute for actual backend used.

    """

    def __init__(
            self,
            n_clusters=8,
            init='k-means++',
            n_init=1,
            max_iter=300,
            tol=1e-4,
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=1,
            algorithm='auto',
            # Beyond sklearn (with optimal defaults)
            gpu_id=0,
            n_gpus=-1,
            do_checks=1,
            backend='auto'):

        import os
        _backend = os.environ.get('H2O4GPU_BACKEND', None)
        if _backend is not None:
            backend = _backend

        # FIXME: Add init as array and kmeans++ to h2o4gpu
        # setup backup to sklearn class
        # (can remove if fully implement sklearn functionality)
        self.do_sklearn = False
        if backend == 'auto':
            example = np.array([1, 2, 3])
            # pylint: disable=unidiomatic-typecheck
            if type(init) == type(example):
                KMeans._print_verbose(
                    verbose, 0,
                    "'init' as ndarray of centers not yet supported."
                    "Running ScikitLearn CPU version.")
                self.do_sklearn = True
            # FIXME: Add n_init to h2o4gpu
            if n_init != 1:
                KMeans._print_verbose(verbose, 0, "'n_init' not supported. "
                                      "Running h2o4gpu with n_init = 1.")
            if precompute_distances != "auto":
                KMeans._print_verbose(verbose, 0,
                                      "'precompute_distances' not used.")
        elif backend == 'sklearn':
            self.do_sklearn = True
        elif backend == 'h2o4gpu':
            self.do_sklearn = False
        if self.do_sklearn:
            self.backend = 'sklearn'
        else:
            self.backend = 'h2o4gpu'

        from h2o4gpu.cluster import k_means_
        self.model_sklearn = k_means_.KMeansSklearn(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            precompute_distances=precompute_distances,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            n_jobs=n_jobs,
            algorithm=algorithm)
        self.model_h2o4gpu = KMeansH2O(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            precompute_distances=precompute_distances,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            n_jobs=n_jobs,
            algorithm=algorithm,
            # H2O4GPU
            gpu_id=gpu_id,
            n_gpus=n_gpus,
            do_checks=do_checks)
        # pylint: disable=protected-access
        if self.do_sklearn or self.model_h2o4gpu._load_lib() is None:
            self.model = self.model_sklearn
            KMeans._print_verbose(verbose, 0, "Using ScikitLearn backend.")
        else:
            self.model = self.model_h2o4gpu
            KMeans._print_verbose(verbose, 0, "Using h2o4gpu backend.")

    def fit(self, X, y=None):
        res = self.model.fit(X, y)
        self.set_attributes()
        return res

    def fit_predict(self, X, y=None):
        res = self.model.fit_predict(X, y)
        self.set_attributes()
        return res

    def fit_transform(self, X, y=None):
        res = self.model.fit_transform(X, y)
        self.set_attributes()
        return res

    def get_params(self, deep=True):
        res = self.model.get_params(deep)
        self.set_attributes()
        return res

    def predict(self, X):
        res = self.model.predict(X)
        self.set_attributes()
        return res

    def score(self, X, y=None):
        # FIXME: Add score to h2o4gpu
        res = self.model_sklearn.score(X, y)
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

    def set_attributes(self):
        s = _setter(oself=self, e1=NameError, e2=AttributeError)

        s('oself.cluster_centers_ = oself.model.cluster_centers_')
        s('oself.labels_ = oself.model.labels_')
        self.inertia_ = None
        s('oself.inertia_ = oself.model.intertia_')

    # TODO use a proper logger in Python classes
    @staticmethod
    def _print_verbose(verbose, level, msg):
        if verbose > level:
            print(msg)
            sys.stdout.flush()
