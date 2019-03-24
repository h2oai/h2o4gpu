import scipy
import numpy as np
import h2o4gpu

if __name__ == '__main__':
    train = scipy.sparse.load_npz('../data/netflix_train.npz').tocsc()
    test = scipy.sparse.load_npz('../data/netflix_test.npz').tocoo()
    print()
    scores = []
    factorization = h2o4gpu.solvers.FactorizationH2O(
        10, 0.001, max_iter=100)
    factorization.fit(train, X_test=test, scores=scores, verbose=True)
