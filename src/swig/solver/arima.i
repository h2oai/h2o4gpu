/* File : arima.i */
%{
#include "../../include/solver/arima.h"
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

%apply (float *IN_ARRAY1) {float* ts_data, float* theta, float* phi};
%apply (double *IN_ARRAY1) {double* ts_data, double* theta, double* phi};

%include "../../include/solver/arima.h"