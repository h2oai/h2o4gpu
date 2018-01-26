%module daicx
%{
  #define SWIG_FILE_WITH_INIT
  #include "../include/metrics/metrics.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double *y, int n), 
                                      (double *yhat, int m), 
                                      (double *w, int l)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *cm, int k, int j)};

%include "../include/metrics/metrics.h"
