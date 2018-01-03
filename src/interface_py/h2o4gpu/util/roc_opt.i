%module roc_opt
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
%include "../include/metrics/metrics.h"
