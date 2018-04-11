/* File : pca.i */
%{
#include "../../include/solver/pca.h"
%}

%rename("params_pca") pca::params;

%apply (double *IN_FARRAY2) {double *_X};
%apply (double *INPLACE_FARRAY2) {double *_Q, double *_U, double *_mean};
%apply (double *INPLACE_ARRAY1) {double *_w, double *_mean, double *_explained_variance, double *_explained_variance_ratio};

%apply (float *IN_FARRAY2) {float *_X};
%apply (float *INPLACE_FARRAY2) {float *_Q, float *_U, float *_mean};
%apply (float *INPLACE_ARRAY1) {float *_w, float *_mean, float *_explained_variance, float *_explained_variance_ratio};

%include "../../include/solver/pca.h"