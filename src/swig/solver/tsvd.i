/* File : tsvd.i */
%{
#include "../../include/solver/tsvd.h"
%}

%rename("params_tsvd") tsvd::params;

%apply (float *IN_FARRAY2) {float *_X};
%apply (float *INPLACE_FARRAY2) {float *_Q, float *_U, float *_X_transformed};
%apply (float *INPLACE_ARRAY1) {float *_w, float *_explained_variance, float *_explained_variance_ratio};

%apply (double *IN_FARRAY2) {double *_X};
%apply (double *INPLACE_FARRAY2) {double *_Q, double *_U, double *_X_transformed};
%apply (double *INPLACE_ARRAY1) {double *_w, double *_explained_variance, double *_explained_variance_ratio};

%include "../../include/solver/tsvd.h"