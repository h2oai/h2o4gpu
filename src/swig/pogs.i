/* File : pogs.i */
%{
#include "../interface_c/h2o4gpu_c_api.h"
%}

%apply (float *IN_ARRAY1) {float *A, float *nzvals};
%apply (double *IN_ARRAY1) {double *A, double *nzvals};

%apply (int *IN_ARRAY1) {int *indices, int *pointers};

%apply (float *INPLACE_ARRAY1) {float *f_a, float *f_b, float *f_c, float *f_d, float *f_e,
                                float *g_a, float *g_b, float *g_c, float *g_d, float *g_e};
%apply (double *INPLACE_ARRAY1) {double *f_a, double *f_b, double *f_c, double *f_d, double *f_e,
                                double *g_a, double *g_b, double *g_c, double *g_d, double *g_e};

%include "../interface_c/h2o4gpu_c_api.h"