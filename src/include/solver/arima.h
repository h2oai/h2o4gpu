#ifndef SRC_INCLUDE_SOLVER_ARIMA_H

void arima_fit_float(const int p, const int d, const int q,
                     const float* ts_data, const int length, float* theta,
                     float* phi);

#endif