import gc
import random
import time
import numpy as np
import pytest
import h2o4gpu
from h2o4gpu.solvers import elastic_net
from h2o4gpu import linear_model
from h2o4gpu.metrics import r2_score
from h2o4gpu.model_selection import train_test_split


def generate_data(nrows, ncols, s=0):

    X = np.random.uniform(-100, 100, size=(nrows, ncols))
    coefs = np.random.randn(ncols)
    const_coef = np.random.randn(1)

    # index of sparse coefficients
    if s != 0:
        zero_coef_loc = random.sample(range(ncols), s)
        coefs[zero_coef_loc] = 0

    # target value
    y = np.dot(X, coefs) + const_coef

    # adding random noise
    sigma = 0.01 * np.std(y)
    y += sigma * np.random.randn(nrows)

    return X, y


def fit_model(X_train, y_train, X_test, y_test, reg_type='enet'):

    if reg_type == 'lasso':
        tol = 1e-2
        alpha = 1.0
        n_threads = None
        n_alphas = 1
        n_lambdas = 1
        n_folds = 1
        lambda_max = alpha
        lambda_min_ratio = 1.0
        lambda_stop_early = False
        store_full_path = 1
        alphas = None
        lambdas = None
        alpha_min = 1.0
        alpha_max = 1.0
        n_gpus = -1
        fit_intercept = True
        max_iter = 5000
        glm_stop_early = True
        glm_stop_early_error_fraction = 1.0
        verbose = False

        reg_h2o = elastic_net.ElasticNetH2O(n_threads=n_threads,
                                            n_gpus=n_gpus,
                                            fit_intercept=fit_intercept,
                                            lambda_min_ratio=lambda_min_ratio,
                                            n_lambdas=n_lambdas,
                                            n_folds=n_folds,
                                            n_alphas=n_alphas,
                                            tol=tol,
                                            lambda_stop_early=lambda_stop_early,
                                            glm_stop_early=glm_stop_early,
                                            glm_stop_early_error_fraction=glm_stop_early_error_fraction,
                                            max_iter=max_iter,
                                            verbose=verbose,
                                            store_full_path=store_full_path,
                                            lambda_max=lambda_max,
                                            alpha_max=alpha_max,
                                            alpha_min=alpha_min,
                                            alphas=alphas,
                                            lambdas=lambdas,
                                            order=None)

        reg_sklearn = linear_model.LassoSklearn()
    elif reg_type == 'ridge':
        reg_h2o = h2o4gpu.Ridge()
        reg_sklearn = linear_model.RidgeSklearn()
    elif reg_type == 'enet':
        reg_h2o = h2o4gpu.ElasticNet()  # update when the wrapper is done
        reg_sklearn = linear_model.ElasticNetSklearn()

    start_h2o = time.time()
    reg_h2o.fit(X_train, y_train, free_input_data=1)
    time_h2o = time.time() - start_h2o

    start_sklearn = time.time()
    reg_sklearn.fit(X_train, y_train)
    time_sklearn = time.time() - start_sklearn

    # Predicting test values
    y_pred_h2o = reg_h2o.predict(X_test, free_input_data=1)
    y_pred_h2o = y_pred_h2o.squeeze()

    y_pred_sklearn = reg_sklearn.predict(X_test)

    # Calculating R^2 scores
    r2_h2o = r2_score(y_test, y_pred_h2o)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)

    # Clearing the memory
    reg_h2o.free_sols()
    reg_h2o.free_preds()
    reg_h2o.finish()
    del reg_h2o
    del reg_sklearn
    gc.collect()

    return time_h2o, time_sklearn, r2_h2o, r2_sklearn

def func():

    n_rows = [600000, 800000]
    n_cols = [400, 600, 800, 1000, 1200]
    for rows in n_rows:
        # res = {}
        # res['n_rows'] = []
        # res['n_cols'] = []
        # res['t_h2o'] = []
        # res['t_sklearn'] = []
        # res['r2_h2o'] = []
        # res['r2_sklearn'] = []

        for cols in n_cols:
            X, y = generate_data(rows, cols)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            print(X_train.nbytes, 'bytes')

            time_h2o, time_sklearn, r2_h2o, r2_sklearn = fit_model(
                X_train, y_train, X_test, y_test, reg_type='lasso')

            # res['n_rows'].append(rows)
            # res['n_cols'].append(cols)
            # res['t_h2o'].append(ret[0])
            # res['t_sklearn'].append(ret[1])
            # res['r2_h2o'].append(ret[2])
            # res['r2_sklearn'].append(ret[3])
            time.sleep(0.1)
        # res = pd.DataFrame(res)
        # res.to_csv("./benchmarks/results_%1.0f.csv" % rows, index=False)

        time.sleep(0.1)
        print('DONE!')

@pytest.mark.skip("WIP")
def test_memory_leak_check(): func()


if __name__ == '__main__':
    test_memory_leak_check()
