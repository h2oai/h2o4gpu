/* File : ffm.i */
%{
#include "../../include/solver/ffm_api.h"
%}

%rename("params_ffm") ffm::Params;

%apply (int *INPLACE_ARRAY1) {int *labels, int *features, int* fields, int *positions,
                              int *labels_v, int *features_v, int* fields_v, int *positions_v};

%apply (float *INPLACE_ARRAY1) {float* values, float* values_v, float *predictions, float *w};
%apply (double *INPLACE_ARRAY1) {double* values, double* values_v, double *predictions, double *w};

%include "../../include/solver/ffm_api.h"