import scipy
import numpy as np
import h2o4gpu
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import argparse

_lib_h2o4gpu = 'h2o4gpu'
_lib_sklearn = 'sklearn'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Factorize netflix prize dataset.')
    parser.add_argument('lib', choices=['h2o4gpu', 'sklearn'], type=str.lower)

    args = parser.parse_args()

    train = scipy.sparse.load_npz('../data/netflix_train.npz').tocoo()
    test = scipy.sparse.load_npz('../data/netflix_test.npz').tocoo()

    n_components = 20

    if args.lib == _lib_h2o4gpu:
        scores = []
        factorization = h2o4gpu.solvers.FactorizationH2O(
            n_components, 0.005, max_iter=85)
        factorization.fit(train, X_test=test, scores=scores, verbose=True)
        user_vec = np.take(factorization.XT, train.row, axis=0)
        movie_vec = np.take(factorization.thetaT, train.col, axis=0)
        features = np.concatenate([user_vec, movie_vec], axis=1)
        print('training boosting model')
        boosting = h2o4gpu.GradientBoostingRegressor(
            verbose=True, criterion='friedman_mse')
        boosting.fit(features, y=train.data)
        pred = boosting.predict(features)
        print('train score: {0}'.format(
            np.sqrt(mean_squared_error(pred, train.data))))

        user_vec = np.take(factorization.XT, test.row, axis=0)
        movie_vec = np.take(factorization.thetaT, test.col, axis=0)
        features = np.stack([user_vec, movie_vec], axis=1)
        pred = boosting.predict(features, y=test.data)
        print('cv score: {0}'.format(
            np.sqrt(mean_squared_error(pred, train.data))))

    else:
        model = NMF(n_components=n_components, init='random',
                    random_state=0, max_iter=100)
        W = model.fit_transform(train)
        H = model.components_.T

        train = train.tocoo()

        a = np.take(W, train.row, axis=0)
        b = np.take(H, train.col, axis=0)
        val = np.sum(a * b, axis=1)

        print(np.sqrt(mean_squared_error(val, train.data)))

        a = np.take(W, test.row, axis=0)
        b = np.take(H, test.col, axis=0)
        val = np.sum(a * b, axis=1)

        print(np.sqrt(mean_squared_error(val, test.data)))
