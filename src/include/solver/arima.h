#ifndef SRC_INCLUDE_SOLVER_ARIMA_H

void arima_fit_float(const int p, const int d, const int q,
                     const float* ts_data, const int length, float* theta,
                     float* phi, const int maxiter);

void arima_fit_double(const int p, const int d, const int q,
                      const double* ts_data, const int length, double* theta,
                      double* phi, const int maxiter);

#endif