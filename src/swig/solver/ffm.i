/* File : ffm.i */
%{
#include "../../include/solver/ffm_api.h"
%}

%rename("params_ffm") ffm::Params;

%apply (int *INPLACE_ARRAY1) {int *labels};
%apply (unsigned long *INPLACE_ARRAY1) {size_t *features, size_t* fields, size_t *positions};

%apply (float *INPLACE_ARRAY1) {float* values, float *scales, float *predictions, float *w};
%apply (double *INPLACE_ARRAY1) {double* values, double *scales, double *predictions, double *w};

%include "../../include/solver/ffm_api.h"