#pragma once

#ifdef WIN32
#define pca_export __declspec(dllexport)
#else
#define pca_export
#endif

namespace pca {

typedef struct params {
  int X_n;
  int X_m;
  int k;
  bool whiten;
  const char *algorithm;
  int verbose;
  int gpu_id;
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

pca_export void pca_float(const float *_X, float *_Q, float *_w, float *_U, float *_explained_variance, float *_explained_variance_ratio, float *_mean, params _param);
pca_export void pca_double(const double *_X, double *_Q, double *_w, double *_U, double *_explained_variance, double *_explained_variance_ratio, double *_mean, params _param);

}
