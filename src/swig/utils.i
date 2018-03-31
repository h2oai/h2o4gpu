/* File : utils.i */
%{
extern int modelfree1_double(double *aptr);
extern int modelfree1_float(float *aptr);
%}

%apply double *IN {double *aptr};
%apply float *IN {float *aptr};

extern int modelfree1_double(double *aptr);
extern int modelfree1_float(float *aptr){;