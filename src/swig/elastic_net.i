/* File : elastic_net.i */
%{
#include "../common/elastic_net_ptr.h"

typedef double** ppdouble;
typedef float** ppfloat;

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        ppdouble a, ppdouble b, ppdouble c, ppdouble d, ppdouble e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       ppfloat a, ppfloat b, ppfloat c, ppfloat d, float **e);

%}
%include "cpointer.i"
%pointer_functions(float, floatp);
%pointer_functions(double, doublep);

%apply (float *IN_ARRAY1) {float *alphas, float *lambdas, float* trainX, float* trainY, float* validX, float* validY, float *weight};
%apply (double *IN_ARRAY1) {double *alphas, double *lambdas, double* trainX, double* trainY, double* validX, double* validY, double *weight};
%apply (float **INPLACE_ARRAY1) {float **Xvsalphalambda, float **Xvsalpha, float **validPredsvsalphalambda, float **validPredsvsalpha};
%apply (double **INPLACE_ARRAY1) {double **Xvsalphalambda, double **Xvsalpha, double **validPredsvsalphalambda, double **validPredsvsalpha};

%apply int *OUTPUT {size_t *countfull, size_t *countshort, size_t *countmore}

typedef double** ppdouble;
typedef float** ppfloat;

%typemap(in) ppfloat {
    $1 = (ppfloat) &$input;
}

%typemap(in) ppdouble {
    $1 = (ppdouble) &$input;
}

%include "../common/elastic_net_ptr.h"

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        ppdouble a, ppdouble b, ppdouble c, ppdouble d, ppdouble e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       ppfloat a, ppfloat b, ppfloat c, ppfloat d, ppfloat e);