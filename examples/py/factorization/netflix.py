import scipy
import numpy as np
import h2o4gpu

if __name__ == '__main__':
    train = scipy.sparse.load_npz('../data/netflix_train.npz').tocsc()
    print(train.shape, train.dtype)
    print(train.nnz)
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        10, 0.001, max_iter=10)
    factorization.fit(train, scores=scores, verbose=True)
