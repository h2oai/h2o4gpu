from ctypes import *
from h2ogpuml.types import cptr
import numpy as np
import time
import sys
from h2ogpuml.libs.kmeans_gpu import GPUlib
from h2ogpuml.libs.kmeans_cpu import CPUlib


class KMeans(object):
    def __init__(self, gpu_id=0, n_gpus=1, k=10, max_iterations=1000, threshold=1E-3, init_from_labels=False,
                 init_labels="randomselect", init_data="randomselect", verbose=0, seed=None):
        self.solver = KMeansBaseSolver(gpu_id, n_gpus, k, max_iterations, threshold,
                                       init_from_labels, init_labels, init_data, verbose, seed)

    def fit(self, X, y):
        return self.solver.fit(X, y)

    def predict(self, X):
        return self.solver.predict(X)

    def sklearnpredict(self, X):
        return self.solver.sklearnpredict(X)

    def transform(self, X):
        return self.solver.transform(X)

    def fit_transform(self, X, y):
        return self.solver.fit_transform(X, y)

    def fit_predict(self, X, y):
        return self.solver.fit_predict(X, y)

    def get_params(self):
        return self.solver.get_params()

    def set_params(self, **kwargs):
        return self.solver.set_params(**kwargs)


class KMeansBaseSolver(object):
    def __init__(self, gpu_id=0, n_gpus=1, k=10, max_iterations=1000, threshold=1E-3, init_from_labels=False,
                 init_labels="randomselect", init_data="randomselect", verbose=0, seed=None):
        self.k = k
        self.gpu_id = gpu_id
        # n_gpus, deviceCount = devicecount(n_gpus=n_gpus)
        deviceCount = 2
        self.n_gpus = n_gpus
        self.deviceCount = deviceCount
        self.max_iterations = max_iterations
        self.init_from_labels = init_from_labels
        self.init_labels = init_labels
        self.init_data = init_data
        self.threshold = threshold
        self.didfit = 0
        self.didsklearnfit = 0
        self.verbose = verbose
        if seed is None:
            import random
            self.seed = random.randint(0, 32000)
        else:
            self.seed = seed

    def get_params(self):
        # input must be sklearn args
        params = {'n_clusters': self.k, 'n_gpus': self.n_gpus, 'max_iterations': self.max_iterations, 'init': 'random',
                  'algorithm': 'auto', 'precompute_distances': True, 'tol': self.threshold, 'n_jobs': -1,
                  'random_state': self.seed, 'verbose': self.verbose, 'copy_x': True}
        return params

    # input must be sklearn args
    def set_params(self, n_clusters=None, n_gpus=None, max_iter=None, init=None, algorithm=None,
                   precompute_distances=None, tol=None, n_jobs=None, random_state=None, verbose=None, copy_x=None):
        if n_clusters is not None:
            if self.verbose > 1:
                print("Changing n_clusters from %d to %d" % (self.k, n_clusters))
                sys.stdout.flush()
            self.k = n_clusters
        if n_gpus is not None:
            self.n_gpus = n_gpus
        if max_iter is not None:
            self.max_iterations = max_iter
        if random_state is not None:
            self.seed = random_state
        if verbose is not None:
            self.verbose = verbose
            # can add more if want to modify

    def KMeansInternal(self, ordin, k, max_iterations, init_from_labels, init_labels, init_data,
                       threshold, mTrain, n, data, labels):
        self.ord = ord(ordin)
        self.k = k
        self.max_iterations = max_iterations
        self.init_from_labels = init_from_labels
        self.init_labels = init_labels
        self.init_data = init_data
        self.threshold = threshold

        if (data.dtype == np.float64):
            if self.verbose > 0:
                print("Detected np.float64 data")
            sys.stdout.flush()
            self.double_precision = 1
            myctype = c_double
            mydtype = np.float64
        elif (data.dtype == np.float32):
            if self.verbose > 0:
                print("Detected np.float32 data")
            sys.stdout.flush()
            self.double_precision = 0
            myctype = c_float
            mydtype = np.float32
        else:
            print("Unknown data type, should be either np.float32 or np.float64")
            print(data.dtype)
            sys.stdout.flush()
            return

        if self.init_from_labels == False:
            c_init_from_labels = 0
        else:
            c_init_from_labels = 1

        if self.init_labels == "random":
            c_init_labels = 0
        elif self.init_labels == "randomselect":
            c_init_labels = 1
        else:
            print("Unknown init_labels %s" % self.init_labels)
            sys.stdout.flush()
            return

        if self.init_data == "random":
            c_init_data = 0
        elif self.init_data == "selectstrat":
            c_init_data = 1
        elif self.init_data == "randomselect":
            c_init_data = 2
        else:
            print("Unknown init_data %s" % self.init_data)
            sys.stdout.flush()
            return

        res = c_void_p(0)
        c_data = cptr(data, dtype=myctype)
        c_labels = cptr(labels, dtype=c_int)
        t0 = time.time()
        #######################
        # set library to use
        lib = self._choose_lib()
        assert lib is not None, "Couldn't instantiate KMeans Library"

        if self.double_precision == 0:
            status = lib.make_ptr_float_kmeans(0, self.verbose, self.seed, self.gpu_id, self.n_gpus, mTrain, n,
                                               c_int(self.ord), self.k,
                                               self.max_iterations, c_init_from_labels, c_init_labels, c_init_data,
                                               self.threshold, c_data, c_labels, None, pointer(res))
        else:
            status = lib.make_ptr_double_kmeans(0, self.verbose, self.seed, self.gpu_id, self.n_gpus, mTrain, n,
                                                c_int(self.ord), self.k,
                                                self.max_iterations, c_init_from_labels, c_init_labels, c_init_data,
                                                self.threshold, c_data, c_labels, None, pointer(res))
        if status:
            raise ValueError('KMeans failed in C++ library')
            sys.stdout.flush()

        self.centroids = np.fromiter(cast(res, POINTER(myctype)), dtype=mydtype, count=self.k * n)
        self.centroids = np.reshape(self.centroids, (self.k, n))

        t1 = time.time()
        return (self.centroids, t1 - t0)

    def _choose_lib(self):
        gpulibgetter = GPUlib()
        gpulib = gpulibgetter.get()
        cpulibgetter = CPUlib()
        cpulib = cpulibgetter.get()
        if ((self.n_gpus == 0) or (gpulib is None) or (self.deviceCount == 0)):
            if self.verbose > 0:
                print("\nUsing CPU KMeans solver\n")
                sys.stdout.flush()
            lib = cpulib
        elif ((self.n_gpus > 0) or (cpulib is None) or (self.deviceCount == 0)):
            if self.verbose > 0:
                print("\nUsing GPU KMeans solver with %d GPUs\n" % self.n_gpus)
                sys.stdout.flush()
            lib = gpulib
        else:
            lib = None
        return lib

    def fit(self, X, y=None):
        import pandas as pd
        self.didfit = 1
        dochecks = 1
        ##########
        if isinstance(X, pd.DataFrame):
            Xnp = X.values
        else:
            Xnp = X
        if isinstance(y, pd.DataFrame):
            ynp = y.values
        else:
            ynp = y
        #########
        if dochecks == 1:
            assert np.isfinite(Xnp).all(), "Xnp contains Inf"
            assert not np.isnan(Xnp).any(), "Xnp contains NA"
        # Xnp = Xnp.astype(np.float32)
        self.Xnp = Xnp
        self.rows = np.shape(Xnp)[0]
        self.cols = np.shape(Xnp)[1]
        ##########
        if ynp is None:
            ynp = np.random.randint(self.rows, size=self.rows) % self.k
        if dochecks == 1:
            assert np.isfinite(ynp).all(), "ynp contains Inf"
            assert not np.isnan(ynp).any(), "ynp contains NA"
        ynp = ynp.astype(np.int)
        ynp = np.mod(ynp, self.k)
        self.ynp = ynp
        ##########
        t0 = time.time()
        if np.isfortran(Xnp):
            self.ord = 'c'
        else:
            self.ord = 'r'
        #
        centroids, timefit0 = self.KMeansInternal(self.ord, self.k, self.max_iterations,
                                                  self.init_from_labels, self.init_labels, self.init_data,
                                                  self.threshold, self.rows, self.cols, Xnp, ynp)
        t1 = time.time()
        if (np.isnan(centroids).any()):
            centroids = centroids[~np.isnan(centroids).any(axis=1)]
            print("Removed " + str(self.k - centroids.shape[0]) + " empty centroids")
            self.k = centroids.shape[0]
        self.centroids = centroids
        return (self.centroids, timefit0, t1 - t0)

    def sklearnfit(self):
        if (self.didsklearnfit == 0):
            self.didsklearnfit = 1
            import sklearn.cluster as SKCluster
            self.model = SKCluster.KMeans(self.k, max_iter=1, init=self.centroids, n_init=1)
            self.model.fit(self.Xnp, self.ynp)

    def sklearnpredict(self, X):
        self.sklearnfit()
        return self.model.predict(X)

    def predict(self, X):
        assert not np.isnan(X).any(), "X contains NA"
        self.prediction = self._predict(X)
        return self.prediction

    def _predict(self, X):
        c_data, _ = self._to_cdata(X)
        c_init_from_labels = 0
        c_init_labels = 0
        c_init_data = 0

        rows = np.shape(X)[0]
        cols = np.shape(X)[1]

        c_centroids, _ = self._to_cdata(self.centroids)
        c_res = c_void_p(0)

        lib = self._choose_lib()
        assert lib is not None, "Couldn't instantiate KMeans Library"

        if self.double_precision == 0:
            lib.make_ptr_float_kmeans(1, self.verbose, self.seed, self.gpu_id, self.n_gpus, rows, cols, c_int(self.ord),
                                      self.k,
                                      self.max_iterations, c_init_from_labels, c_init_labels, c_init_data,
                                      self.threshold, c_data, None, c_centroids, pointer(c_res))
        else:
            lib.make_ptr_double_kmeans(1, self.verbose, self.seed, self.gpu_id, self.n_gpus, rows, cols,
                                       c_int(self.ord), self.k,
                                       self.max_iterations, c_init_from_labels, c_init_labels, c_init_data,
                                       self.threshold, c_data, None, c_centroids, pointer(c_res))

        preds = np.fromiter(cast(c_res, POINTER(c_int)), dtype=np.int32, count=rows)
        preds = np.reshape(preds, rows)
        return preds

    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        L = np.mod(y, self.k)
        self.fit(X, L)
        return self.transform(X)

    def fit_predict(self, X, y):
        L = np.mod(y, self.k)
        self.fit(X, L)
        return self.predict(X)
        # FIXME TODO: Still need (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict)
        # get_params, score, set_params
        # various parameters like init, algorithm, n_init
        # need to ensure output as desired

    def _to_cdata(self, data):
        if data.dtype == np.float64:
            print("Detected np.float64 data")
            sys.stdout.flush()
            self.double_precision = 1
            myctype = c_double
        elif data.dtype == np.float32:
            print("Detected np.float32 data")
            sys.stdout.flush()
            self.double_precision = 0
            myctype = c_float
        else:
            print("Unknown data type, should be either np.float32 or np.float64")
            print(data.dtype)
            sys.stdout.flush()
            return None, None
        return cptr(data, dtype=myctype), myctype
