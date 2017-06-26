from h2oaiglm.libs.kmeans_gpu import h2oaiglmKMeansGPU
from h2oaiglm.solvers.kmeans_base import KMeansBaseSolver
from ctypes import *

if not h2oaiglmKMeansGPU:
    print('\nWarning: Cannot create a H2OAIKMeans GPU Solver instance without linking Python module to a compiled H2OAIGLM GPU library')
    print('> Setting h2oaiglm.KMeansGPU=None')
    print('> Add CUDA libraries to $PATH and re-run setup.py\n\n')
    KMeansSolverGPU=None
else:
    class KMeansSolverGPU(object):
        def __init__(self, gpu_id, n_gpus, k, max_iterations, threshold, init_from_labels, init_labels, init_data):
            self.solver = KMeansBaseSolver(h2oaiglmKMeansGPU, gpu_id, n_gpus, k, max_iterations, threshold, init_from_labels, init_labels, init_data)

        def KMeansInternal(self, gpu_id, n_gpu, ordin, k, max_iterations, init_from_labels, init_labels, init_data, threshold, mTrain, n, data, labels):
            return self.solver.KMeansInternal(gpu_id, n_gpu, ordin, k, max_iterations, init_from_labels, init_labels, init_data, threshold, mTrain, n, data, labels)
        def fit(self, X, L):
            return self.solver.fit(X,L)
        def sklearnfit(self):
            return self.solver.sklearnfit()
        def predict(self, X):
            return self.solver.predict(X)
        def transform(self, X):
            return self.solver.transform(X)
        def fit_transform(self, X, origL):
            return self.solver.fit_transform(X,origL)
        def fit_predict(self, X, origL):
            return self.solver.fit_predict(X,origL)
        
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
    
