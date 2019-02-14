/* File : matrix_dense.i */
%{
extern int modelfree1_float(float *aptr);
extern int modelfree1_double(double *aptr);

extern int modelfree2_float(float *aptr);
extern int modelfree2_double(double *aptr);
%}

%apply float *INPUT {float *aptr}
%apply double *INPUT {double *aptr}

extern int modelfree1_float(float *aptr);
extern int modelfree1_double(double *aptr);

extern int modelfree2_float(float *aptr);
extern int modelfree2_double(double *aptr);
