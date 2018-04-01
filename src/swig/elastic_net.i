/* File : elastic_net.i */
%{
#include "../common/elastic_net_ptr.h"

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        double**a, double**b, double**c, double**d, double **e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       float**a, float**b, float**c, float**d, float **e);

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

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        double**a, double**b, double**c, double**d, double **e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       float**a, float**b, float**c, float**d, float **e);