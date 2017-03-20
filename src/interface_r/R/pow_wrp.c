#include "R.h"

#include "Rdefines.h"

	#include "Rinternals.h"

	#include "mkl.h"

SEXP mkl_vdpow(SEXP N, SEXP X, SEXP Y)

	{

	  SEXP Z;

	  int *n;

	  int nn;

	  double *x, *y, *z;

	 

	  /* Convert input arguments to C types */

	  PROTECT(N = AS_INTEGER(N));

	  PROTECT(X = AS_NUMERIC(X));

	  PROTECT(Y = AS_NUMERIC(Y));

	 

	  n = INTEGER_POINTER(N);

	  x = NUMERIC_POINTER(X);

	  y = NUMERIC_POINTER(Y);

	 

	  /* Allocate memory to store results */

	  nn = *n;

	  PROTECT(Z = NEW_NUMERIC(nn));

	  z = NUMERIC_POINTER(Z);

	 

	  vdPow(nn, x, y, z);

	 

	  UNPROTECT(4);

	 

	  return Z;

	}
