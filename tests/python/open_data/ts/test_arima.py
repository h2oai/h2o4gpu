import pytest
import numpy as np
import h2o4gpu
from scipy import signal
import statsmodels as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA


def predict(y, phi, theta):
    a = np.ones((1 + theta.shape[0],), dtype=y.dtype)
    b = np.ones((1 + phi.shape[0],), dtype=y.dtype)
    a[1:] = theta
    b[1:] = - phi

    residual = signal.lfilter(b, a, y.copy())
    return residual


def score(y, phi, theta):
    res = predict(y, phi, theta)
    return np.sqrt(np.sum(res ** 2)) / y.shape[0]


def validate_arima_generated(p, q, double, arparams, maparams):
    np.random.seed(0)
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    y = arma_generate_sample(ar, ma, 5000)
    model = sm.tsa.arima_model.ARMA(y, (p, q)).fit()
    statsmodels_score = score(y, model.arparams, model.maparams)

    print(model.arparams, model.maparams)

    model = h2o4gpu.solvers.ARIMA(p, 0, q,  double_precision=double)
    model.fit(y)

    h2o_score = score(y, model.phi_, model.theta_)
    print(model.phi_, model.theta_)
    print(statsmodels_score, h2o_score)
    assert np.allclose(statsmodels_score, h2o_score, 1e-4, 1e-5)


@pytest.mark.parametrize("double", [True, False])
def test_arima_size_2(double):
    validate_arima_generated(2, 2, double, np.array(
        [.75, -.25]), np.array([.65, .35]))


@pytest.mark.parametrize("double", [True, False])
def test_arima_size_3(double):
    validate_arima_generated(3, 3, double, np.array(
        [.75, -.25, -0.1]), np.array([.65, -.15, 0.4]))


if __name__ == '__main__':
    pass
    # test_arima_size_3(True)
