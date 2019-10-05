import numpy as np
import pandas as pd
import h2o4gpu
from scipy import signal
from pmdarima.arima import ARIMA


def predict(y, phi, theta):
    a = np.ones((1 + theta.shape[0],), dtype=y.dtype)
    b = np.ones((1 + phi.shape[0],), dtype=y.dtype)
    a[1:] = theta
    b[1:] = - phi
    # print(b)
    # lfilter
    # a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
    #                   - a[1]*y[n-1] - ... - a[N]*y[n-N]

    residual = signal.lfilter(b, a, y.copy())
    return residual
    # return residual[max(phi.shape[0], theta.shape[0]):]


def test_arima():
    p = 10
    d = 0
    q = 8
    model = h2o4gpu.solvers.ARIMA(p, d, q)
    ts = pd.read_csv(
        'open_data/ts/hourly-energy-consumption/PJME_hourly.csv')
    y = ts['PJME_MW'].values.astype(np.float32)[0:1000]
    y -= np.mean(y)
    model.fit(y, maxiter=200)
    print(model.phi_, model.theta_)
    # print(y)

    r = predict(y, model.phi_, model.theta_)
    # # print(r)
    print(np.sqrt(np.sum(r ** 2)) / y.shape[0])

    model2 = ARIMA((p, d, q))
    model2.fit(y, maxiter=500)
    print(model2.arparams())
    print(model2.maparams())

    r = predict(y, model2.arparams(), model2.maparams())
    # print(r)
    print(np.sqrt(np.sum(r ** 2)) / y.shape[0])


if __name__ == '__main__':
    test_arima()
