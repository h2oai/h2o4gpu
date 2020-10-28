import pytest
import numpy as np
import h2o4gpu
from scipy import signal
import statsmodels as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima_model import ARMA
np.random.seed(0)


def differencing(x, order):
    if order == 0:
        return x
    else:
        return differencing(x[1:] - x[:-1], order - 1)


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


def validate_arma_generated(p, q, double, arparams, maparams):
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
def test_arma_size_2(double):
    validate_arma_generated(2, 2, double, np.array(
        [.75, -.25]), np.array([.65, .35]))


@pytest.mark.parametrize("double", [True, False])
def test_arma_size_3(double):
    validate_arma_generated(3, 3, double, np.array(
        [.75, -.25, -0.1]), np.array([.65, -.15, 0.4]))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("double", [True, False])
def test_arima(order, double):
    p = 3
    q = 3
    ar = np.array([.75, -.25, -0.1])
    ma = np.array([.65, -.15, 0.4])

    ar = np.r_[1, -ar]  # add zero-lag and negate
    ma = np.r_[1, ma]  # add zero-lag
    y = arma_generate_sample(ar, ma, 5000)

    model1 = h2o4gpu.solvers.ARIMA(p, order, q, double)
    model1.fit(y)
    print(model1.phi_, model1.theta_)

    y = differencing(y, order)
    model2 = h2o4gpu.solvers.ARIMA(p, 0, q, double)
    model2.fit(y)

    print(model2.phi_, model2.theta_)

    rtol=2e-4
    atol=2e-7
    if not double:
        rtol=1e-3
        atol=1e-5

    assert np.allclose(model1.phi_, model2.phi_, rtol=rtol, atol=atol)
    assert np.allclose(model1.theta_, model2.theta_, rtol=rtol, atol=atol)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("double", [True, False])
def test_memory_leak(order, double, iterations=100):
    p = 3
    q = 3
    ar = np.array([.75, -.25, -0.1])
    ma = np.array([.65, -.15, 0.4])

    ar = np.r_[1, -ar]  # add zero-lag and negate
    ma = np.r_[1, ma]  # add zero-lag
    y = arma_generate_sample(ar, ma, 5000)

    for _ in range(iterations):
        model2 = h2o4gpu.solvers.ARIMA(p, order, q, double)
        model2.fit(y)


if __name__ == '__main__':
    pass
    # test_arima_size_3(True)
    # test_arima(5, False)
    # test_memory_leak(4, False, 10000)
