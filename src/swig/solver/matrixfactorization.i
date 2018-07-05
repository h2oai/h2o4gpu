/* File: matrixfactorization.i */
%{
#include "../../include/solver/matrixfactorization.h"
%}

%apply (float *INPLACE_FARRAY2) {float *thetaTHost, float *XTHost};
 
#include "../../include/solver/matrixfactorization.h"