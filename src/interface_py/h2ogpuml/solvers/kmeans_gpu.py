# GPU-specific classes for KMeans
class KMeansGPU2():
    def __init__(self, k = 10, **params):
        self.k = k
        self.params = params
        self.didfit=0
        self.didsklearnfit=0
    def fit(self, X):
        import numpy as np
        self.didfit=1
        assert np.isfinite(X).all(), "X contains Inf"
        assert not np.isnan(X).any(), "X contains NA"
        #X = X.astype(np.float32)
        self.X=X
        self.params['average_distance'] = True
        import libKMCUDA ## git clone http://github.com/h2ogpuml/kmcuda/ ; cd kmcuda/src ; cmake -DCMAKE_BUILD_TYPE=Release . ; export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ; python setup.py sdist bdist_wheel ; pip install dist/libKMCUDA-6.2.0-cp36-cp36m-linux_x86_64.whl --upgrade
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
            import sklearn.cluster as SKCluster
            self.model = SKCluster.KMeans(self.k, max_iter=1, init=self.centroids, n_init=1)
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
    
