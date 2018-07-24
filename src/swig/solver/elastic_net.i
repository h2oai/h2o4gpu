/* File : elastic_net.i */
%{
#include "../../common/elastic_net_ptr.h"

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        double** a, double** b, double** c, double** d, double** e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       float** a, float** b, float** c, float** d, float** e);

%}

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

%typemap(in) float *trainXptr,
             float *trainYptr,
             float *validXptr,
             float *validYptr,
             float *weightptr {
    if($input >= 0) {
        $1 = (float *)PyLong_AsVoidPtr($input);
    } else {
        $1 = NULL;
    }
}

%typemap(in) double *trainXptr,
             double *trainYptr,
             double *validXptr,
             double *validYptr,
             double *weightptr {
    if($input >= 0) {
        $1 = (double *)PyLong_AsVoidPtr($input);
    } else {
        $1 = NULL;
    }
}

%apply (float *IN_ARRAY1) {float *alphas, float *lambdas, float* trainX, float* trainY, float* validX, float* validY, float *weight};
%apply (double *IN_ARRAY1) {double *alphas, double *lambdas, double* trainX, double* trainY, double* validX, double* validY, double *weight};

%apply size_t *INOUT {size_t *countfull, size_t *countshort, size_t *countmore}

%include "../../common/elastic_net_ptr.h"

extern int make_ptr_double(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                        const double* trainX, const double* trainY, const double* validX, const double* validY, const double *weight,
                        double** a, double** b, double** c, double** d, double** e);
extern int make_ptr_float(int sharedA, int sourceme, int sourceDev, size_t mTrain, size_t n, size_t mValid, const char ord,
                       const float* trainX, const float* trainY, const float* validX, const float* validY, const float *weight,
                       float** a, float** b, float** c, float** d, float** e);