/* File : pca.i */
%{
#include "../../gpu/pca/pca.h"
%}

%rename("params_pca") pca::params;

%apply (double *IN_FARRAY2) {double *_X};
%apply (double *INPLACE_FARRAY2) {double *_Q, double *_U, double *_mean};
%apply (double *INPLACE_ARRAY1) {double *_w, double *_mean, double *_explained_variance, double *_explained_variance_ratio};

%include "../../gpu/pca/pca.h"