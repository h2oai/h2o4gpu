/* File : factorization.i */
%{
#include "../../include/solver/factorization.h"
%}

%typemap(in) int** (int* tmp) {
    if ((SWIG_ConvertPtr($input, (void **) &tmp, $*1_descriptor, $disown | 0)) == -1) {
        tmp = NULL;
    }
    $1 = &tmp;
}
%typemap(argout) int** {
    %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(in) float** (float* tmp) {
    if ((SWIG_ConvertPtr($input, (void **) &tmp, $*1_descriptor, $disown | 0)) == -1) {
        tmp = NULL;
    }
    $1 = &tmp;
}
%typemap(argout) float** {
    %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(in) double** (double* tmp) {
    if ((SWIG_ConvertPtr($input, (void **) &tmp, $*1_descriptor, $disown | 0)) == -1) {
        tmp = NULL;
    }
    $1 = &tmp;
}
%typemap(argout) double** {
    %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%apply (int *IN_ARRAY1) {int* csrRowIndexHostPtr, int* csrColIndexHostPtr, int* cscRowIndexHostPtr,
                         int* cscColIndexHostPtr, int* cooRowIndexHostPtr, int* cooColIndexHostPtr,
                         int* cooRowIndexTestHostPtr, int* cooColIndexTestHostPtr};
%apply (float *IN_ARRAY1) {float* csrValHostPtr, float* cscValHostPtr, float* cooValHostPtr, float* thetaTHost, float* XTHost, float* cooValTestHostPtr};
%apply (double *IN_ARRAY1) {double* csrValHostPtr, double* cscValHostPtr, double* cooValHostPtr, double* thetaTHost, double* XTHost, double* cooValTestHostPtr};

%include "../../include/solver/factorization.h"
