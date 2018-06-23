#pragma once

#ifdef WIN32
#define tsvd_export __declspec(dllexport)
#else
#define tsvd_export
#endif

namespace matrix {

template<typename T>
class Matrix;

}

namespace device {
class DeviceContext;

}

namespace tsvd {

typedef struct params {
  int X_n;
  int X_m;
  int k;
  char *algorithm;
  int n_iter;
  int random_state;
  float tol;
  int verbose;
  int gpu_id;
  bool whiten;
} params;

/**
 *
 * \param 		  	_X
 * \param [in,out]	_Q
 * \param [in,out]	_w
 * \param [in,out]  _U
 * \param [out] 	_explained_variance
 * \param[out]		_explained_variance_ratio
 * \param 		  	_param
 */

tsvd_export void truncated_svd_float(const float *_X, float *_Q, float *_w, float *_U, float *_X_transformed, float *_explained_variance, float *_explained_variance_ratio, params _param);
tsvd_export void truncated_svd_double(const double *_X, double *_Q, double *_w, double *_U, double *_X_transformed, double *_explained_variance, double *_explained_variance_ratio, params _param);

template<typename T, typename S>
void cusolver_tsvd(matrix::Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param);

template<typename T, typename S>
void power_tsvd(matrix::Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param);

template<typename T, typename S>
tsvd_export void truncated_svd_matrix(matrix::Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param);

}
