/* File : tsvd.i */
%{
#include "../gpu/tsvd/tsvd.h"
%}

%rename("params_tsvd") tsvd::params;

%apply (float *IN_FARRAY2) {float *_X};
%apply (float *INPLACE_FARRAY2) {float *_Q, float *_U};
%apply (float *INPLACE_ARRAY1) {float *_w};

%apply (double *IN_FARRAY2) {double *_X};
%apply (double *INPLACE_FARRAY2) {double *_Q, double *_U};
%apply (double *INPLACE_ARRAY1) {double *_w};

%include "../gpu/tsvd/tsvd.h"
