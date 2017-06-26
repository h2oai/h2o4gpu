from h2oaiglm.libs.kmeans_gpu import h2oaiKMeansGPU
from ctypes import *
from h2oaiglm.types import ORD, cptr
import numpy as np
import time
import sys
from sklearn.cluster import KMeans

if not h2oaiKMeansGPU:
    print('\nWarning: Cannot create a H2OAIKMeans GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
    print('> Setting h2oaiglm.KMeansGPU=None')
    print('> Add CUDA libraries to $PATH and re-run setup.py\n\n')
    KMeansGPU=None
else:
    class KMeansGPUinternal(object):
        def __init__(self, gpu_id, n_gpu, ordin, k, max_iterations, init_from_labels, init_labels, init_data, threshold):
            self.gpu_id = gpu_id
            self.n_gpu = n_gpu
            self.ord = ord(ordin)
            self.k = k
            self.max_iterations=max_iterations
            self.init_from_labels=init_from_labels
            self.init_labels=init_labels
            self.init_data=init_data
            self.threshold=threshold

        def fit(self, mTrain, n, data, labels):

            if (data.dtype==np.float64):
                print("Detected np.float64 data");sys.stdout.flush()
                self.double_precision=1
                myctype=c_double
            if (data.dtype==np.float32):
                print("Detected np.float32 data");sys.stdout.flush()
                self.double_precision=0
                myctype=c_float

            if self.init_from_labels==False:
                c_init_from_labels=0
            elif self.init_from_labels==True:
                c_init_from_labels=1
                
            if self.init_labels=="random":
                c_init_labels=0
            elif self.init_labels=="randomselect":
                c_init_labels=1

            if self.init_data=="random":
                c_init_data=0
            elif self.init_data=="selectstrat":
                c_init_data=1
            elif self.init_data=="randomselect":
                c_init_data=2

            res = c_void_p(0)
            c_data = cptr(data,dtype=myctype)
            c_labels = cptr(labels,dtype=c_int)
            t0 = time.time()
            if self.double_precision==0:
                h2oaiKMeansGPU.make_ptr_float_kmeans(self.gpu_id, self.n_gpu, mTrain, n, c_int(self.ord), self.k, self.max_iterations, c_init_from_labels, c_init_labels, c_init_data, self.threshold, c_data, c_labels, pointer(res))
                self.centroids=np.fromiter(cast(res, POINTER(myctype)), dtype=np.float32, count=self.k*n)
                self.centroids=np.reshape(self.centroids,(self.k,n))
            else:
                h2oaiKMeansGPU.make_ptr_double_kmeans(self.gpu_id, self.n_gpu, mTrain, n, c_int(self.ord), self.k, self.max_iterations, c_init_from_labels, c_init_labels, c_init_data, self.threshold, c_data, c_labels, pointer(res))
                self.centroids=np.fromiter(cast(res, POINTER(myctype)), dtype=np.float64, count=self.k*n)
                self.centroids=np.reshape(self.centroids,(self.k,n))

            t1 = time.time()
            return(self.centroids, t1-t0)

        
    class KMeansGPU(object):
        def __init__(self, gpu_id=0, n_gpus=1, k = 10, max_iterations=1000, threshold=1E-3, init_from_labels=False, init_labels="randomselect", init_data="randomselect", **params):
            self.k = k
            self.gpu_id = gpu_id
            self.n_gpus = n_gpus
            self.params = params
            self.max_iterations=max_iterations
            self.init_from_labels=init_from_labels
            self.init_labels=init_labels
            self.init_data=init_data
            self.threshold=threshold
            self.didfit=0
            self.didsklearnfit=0
        def fit(self, X, L):
            self.didfit=1
            dochecks=1
            if dochecks==1:
                assert np.isfinite(X).all(), "X contains Inf"
                assert not np.isnan(X).any(), "X contains NA"
                assert np.isfinite(L).all(), "L contains Inf"
                assert not np.isnan(L).any(), "L contains NA"
            #X = X.astype(np.float32)
            L = L.astype(np.int)
            L=np.mod(L,self.k)
            self.X=X
            self.L=L
            self.rows=np.shape(X)[0]
            self.cols=np.shape(X)[1]
            self.params['average_distance'] = True
            t0 = time.time()
            if np.isfortran(X):
                self.ord='c'
            else:
                self.ord='r'
            #
            centroids, timefit0 = KMeansGPUinternal(self.gpu_id, self.n_gpus, self.ord, self.k, self.max_iterations, self.init_from_labels, self.init_labels, self.init_data, self.threshold).fit(self.rows,self.cols,X,L)
            t1 = time.time()
            if (np.isnan(centroids).any()):
                centroids = centroids[~np.isnan(centroids).any(axis=1)]
                print("Removed " + str(self.k - centroids.shape[0]) + " empty centroids")
                self.k = centroids.shape[0]
            self.centroids = centroids
            return(self.centroids, timefit0, t1-t0)
        def sklearnfit(self):
            if(self.didsklearnfit==0):
                self.didsklearnfit=1
                self.model = KMeans(self.k, max_iter=1, init=self.centroids, n_init=1)
                self.model.fit(self.X,self.L)
        def predict(self, X):
            self.sklearnfit()
            return self.model.predict(X)
            # no other choice FIXME TODO
        def transform(self, X):
            self.sklearnfit()
            return self.model.transform(X)
            # no other choice FIXME TODO
        def fit_transform(self, X, origL):
            L=np.mod(origL,self.k)
            self.fit(X,L)
            return self.transform(X)
        def fit_predict(self, X, origL):
            L=np.mod(origL,self.k)
            self.fit(X,L)
            return self.predict(X)
        # FIXME TODO: Still need (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit_predict)
        # get_params, score, set_params
        # various parameters like init, algorithm, n_init
        # need to ensure output as desired

        #         src-d (alternative vendor) CUDA K-Means (multi-GPU, but restricted to k%n_gpu=0
    class KMeansGPU2():
        def __init__(self, k = 10, **params):
            self.k = k
            self.params = params
            self.didfit=0
            self.didsklearnfit=0
        def fit(self, X):
            self.didfit=1
            assert np.isfinite(X).all(), "X contains Inf"
            assert not np.isnan(X).any(), "X contains NA"
            #X = X.astype(np.float32)
            self.X=X
            self.params['average_distance'] = True
            import libKMCUDA ## git clone http://github.com/h2oai/kmcuda/ ; cd kmcuda/src ; cmake -DCMAKE_BUILD_TYPE=Release . ; export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ; python setup.py sdist bdist_wheel ; pip install dist/libKMCUDA-6.2.0-cp36-cp36m-linux_x86_64.whl --upgrade
            centroids, assignments, avg_distance = libKMCUDA.kmeans_cuda(X, self.k, **self.params)
            centroids = centroids[~np.isnan(centroids).any(axis=1)]
            #print("Removed " + str(self.k - centroids.shape[0]) + " empty centroids")
            self.k = centroids.shape[0]
            self.centroids = centroids
            self.assignments = assignments     ### NOT CURRENTLY USED
            self.avg_distance = avg_distance   ### NOT CURRENTLY USED
            return(self.centroids)
        def sklearnfit(self):
            if(self.didsklearnfit==0):
                self.didsklearnfit=1
                self.model = KMeans(self.k, max_iter=1, init=self.centroids, n_init=1)
                self.model.fit(self.X)
        def predict(self, X):
            self.sklearnfit()
            return self.model.predict(X)
            # no other choice FIXME TODO
        def transform(self, X):
            self.sklearnfit()
            return self.model.transform(X)
            # no other choice FIXME TODO
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
        def fit_predict(self, X):
            self.fit(X)
            return self.predict(X)
    
