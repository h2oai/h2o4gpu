/* File : ffm.i */
%{
#include "../../include/solver/ffm_api.h"
%}

%rename("params_ffm") ffm::Params;

%apply (int *INPLACE_ARRAY1) {int *labels, int *features, int* fields, int *positions};

%apply (float *INPLACE_ARRAY1) {float* values, float *predictions, float *w};
%apply (double *INPLACE_ARRAY1) {double* values, double *predictions, double *w};

%include "../../include/solver/ffm_api.h"