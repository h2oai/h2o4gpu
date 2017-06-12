from h2oaiglm.libs.kmeans_gpu import h2oaiKMeansGPU
from ctypes import *
from h2oaiglm.types import ORD, cptr
import numpy as np

if not h2oaiKMeansGPU:
    print('\nWarning: Cannot create a H2OAIKMeans GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
    print('> Setting h2oaiglm.KMeansGPU=None')
    print('> Add CUDA libraries to $PATH and re-run setup.py\n\n')
    KMeansGPU=None
else:
    # class KMeansGPUSolver(object):
    #     def __init__(self, lib, nGPUs, ordin, k):
    #         assert lib and (lib==h2oaiKMeansGPU)
    #     self.lib=lib
    #     self.nGPUs=nGPUs
    #     self.sourceDev=0 # assume Dev=0 is source of data for upload_data
    #     self.ord=ord(ordin)
    #     self.k=k

    # def upload_data(self, sourceDev, trainX):
    #     mTrain = trainX.shape[0]
    #     n = trainX.shape[1]
    #     a = c_void_p(0)
    #     if (trainX.dtype==np.float64):
    #         print("Detected np.float64");sys.stdout.flush()
    #         self.double_precision=1
    #         A = cptr(trainX,dtype=c_double)
    #         status = self.lib.make_ptr_double_kmeans(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_int(self.ord), A, pointer(a))
    #
    #     elif (trainX.dtype==np.float32):
    #         print("Detected np.float32");sys.stdout.flush()
    #         self.double_precision=0
    #         A = cptr(trainX,dtype=c_float)
    #         status = self.lib.make_ptr_float_kmeans(c_int(sourceDev), c_size_t(mTrain), c_size_t(n), c_int(self.ord), A, pointer(a))
    #     else:
    #         print("Unknown numpy type detected")
    #         print(trainX.dtype)
    #         sys.stdout.flush()
    #         exit(1)
    #
    #     assert status==0, "Failure uploading the data"
    #     print(a)
    #     return a
    #
    # # sourceDev here because generally want to take in any pointer, not just from our test code
    # def fit(self, sourceDev, mTrain, n, precision, a):
    #     # not calling with self.sourceDev because want option to never use default but instead input pointers from foreign code's pointers
    #     if hasattr(self, 'double_precision'):
    #         whichprecision=self.double_precision
    #     else:
    #         whichprecision=precision
    #     #
    #     if (whichprecision==1):
    #         print("double precision fit")
    #         self.lib.kmeans_ptr_double(
    #             c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
    #             c_size_t(mTrain), c_size_t(n), c_size_t(mValid),c_int(self.intercept), c_int(self.standardize),
    #             c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
    #             a, b, c, d, e)
    #     else:
    #         print("single precision fit")
    #         self.lib.kmeans_ptr_float(
    #             c_int(sourceDev), c_int(1), c_int(self.sharedA), c_int(self.nThreads), c_int(self.nGPUs),c_int(self.ord),
    #             c_size_t(mTrain), c_size_t(n), c_size_t(mValid), c_int(self.intercept), c_int(self.standardize),
    #             c_double(self.lambda_min_ratio), c_int(self.n_lambdas), c_int(self.n_folds), c_int(self.n_alphas),
    #             a, b, c, d, e)
    #     print("Done with fit")



    class KMeansGPUinternal(object):
        def __init__(self, nGPUs, ordin, k, max_iterations, threshold):
            self.nGPUs = nGPUs
            self.ord = ord(ordin)
            self.k = k
            self.max_iterations=max_iterations
            self.threshold=threshold
            
                #KMeansGPUSolver(h2oaiKMeansGPU, nGPUs, ord, k)

        # def upload_data_ptr(self, sourceDev, trainX):
        # 	return self.solver.upload_data(sourceDev, trainX)
        #
        # def fit_ptr(self, sourceDev, mTrain, n, a):
        # 	return self.solver.fit(sourceDev, mTrain, n, a)

        def fit(self, mTrain, n, data, labels):
            res = c_void_p(0)
            c_data = cptr(data,dtype=c_float)
            c_labels = cptr(labels,dtype=c_int)
            h2oaiKMeansGPU.make_ptr_float_kmeans(self.nGPUs, mTrain, n, c_int(self.ord), self.k, self.max_iterations, self.threshold, c_data, c_labels, pointer(res))
            self.centroids=np.fromiter(cast(res, POINTER(c_float)), dtype=np.float32, count=self.k*n)
            self.centroids=np.reshape(self.centroids,(self.k,n))
            return(self.centroids)

        
    class KMeansGPU(object):
        def __init__(self, n_gpus=1, k = 10, max_iterations=1000, threshold=1E-3, **params):
            self.k = k
            self.n_gpus = n_gpus
            self.params = params
            self.max_iterations=max_iterations
            self.threshold=threshold
        def fit(self, X, L):
            dochecks=1
            if dochecks==1:
                assert np.isfinite(X).all(), "X contains Inf"
                assert not np.isnan(X).any(), "X contains NA"
                assert np.isfinite(L).all(), "L contains Inf"
                assert not np.isnan(L).any(), "L contains NA"
            X = X.astype(np.float32)
            L = L.astype(np.int)
            self.rows=np.shape(X)[0]
            self.cols=np.shape(X)[1]
            self.params['average_distance'] = True
            centroids = KMeansGPUinternal(self.n_gpus, 'r', self.k, self.max_iterations, self.threshold).fit(self.rows,self.cols,X,L)
            if (np.isnan(centroids).any()):
                centroids = centroids[~np.isnan(centroids).any(axis=1)]
                print("Removed " + str(self.k - centroids.shape[0]) + " empty centroids")
                self.k = centroids.shape[0]
            self.centroids = centroids
            dosklearnfit=1
            if dosklearnfit==1: # only required because of predict and transform
                from sklearn.cluster import KMeans
                self.model = KMeans(self.k, max_iter=1, init=self.centroids, n_init=1)
                self.model.fit(X,L)
        def predict(self, X):
            dosklearnpredict=1
            if dosklearnpredict==1:
                return self.model.predict(X)
            # no other choice FIXME TODO
        def transform(self, X):
            dosklearntransform=1
            if dosklearntransform==1:
                return self.model.transform(X)
            # no other choice FIXME TODO
        def fit_transform(self, X, origL):
            L=np.mod(origL,self.k)
            self.L=L
            self.fit(X,self.L)
            return self.transform(X)
        def fit_predict(self, X, origL):
            L=np.mod(origL,self.k)
            self.L=L
            self.fit(X,self.L)
            return self.predict(X)
        # FIXME TODO: Still need (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict)
        # get_params, score, set_params
        # various parameters like init, algorithm, n_init
        # need to ensure output as desired
