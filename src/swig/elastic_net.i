/* File : pogs.i */
%{
#include "../common/elastic_net_ptr.h"
%}

%apply (float *IN_ARRAY1) {float *alphas, float *lambdas, float *trainXptr, float *trainYptr, float *validXptr, float *validYptr, float *weightptr};
%apply (double *IN_ARRAY1) {double *alphas, double *lambdas, double *trainXptr, double *trainYptr, double *validXptr, double *validYptr, double *weightptr};
%apply (float **INPLACE_ARRAY1) {float **Xvsalphalambda, float **Xvsalpha, float **validPredsvsalphalambda, float **validPredsvsalpha};
%apply (double **INPLACE_ARRAY1) {double **Xvsalphalambda, double **Xvsalpha, double **validPredsvsalphalambda, double **validPredsvsalpha};

%apply int *OUTPUT {size_t *countfull, size_t *countshort, size_t *countmore}

%apply (float *IN_ARRAY1) {float* trainX, float* trainY, float* validX, float* validY, float *weight};
%apply (double *IN_ARRAY1) {double* trainX, double* trainY, double* validX, double* validY, double *weight};
%apply (float **INPLACE_ARRAY1) {float**a, float**b, float**c, float**d, float **e};
%apply (double **INPLACE_ARRAY1) {double**a, double**b, double**c, double**d, double **e};

%include "../common/elastic_net_ptr.h"