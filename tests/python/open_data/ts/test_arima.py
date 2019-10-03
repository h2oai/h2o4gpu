import numpy as np
import pandas as pd
import h2o4gpu


def test_arima():
    model = h2o4gpu.solvers.ARIMA(10, 0, 10)
    ts = pd.read_csv(
        'open_data/ts/hourly-energy-consumption/PJME_hourly.csv')
    y = ts['PJME_MW'].values.astype(np.float32)
    print(model.phi_, model.theta_)
    model.fit(y)
    print(model.phi_, model.theta_)
    print(y)


if __name__ == '__main__':
    test_arima()
