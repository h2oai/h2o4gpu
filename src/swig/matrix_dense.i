/* File : matrix_dense.i */
%{
extern int modelfree1_float(float *aptr);
extern int modelfree1_double(double *aptr);
%}

%apply double *IN_ARRAY1 {double *aptr};
%apply float *IN_ARRAY1 {float *aptr};

extern int modelfree1_float(float *aptr);
extern int modelfree1_double(double *aptr);