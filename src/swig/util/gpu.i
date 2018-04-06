/* File : gpu.i */
%{
extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern int cudaresetdevice_bare(void);
%}

%apply int *OUTPUT {int *major, int *minor, int *ratioperf}

extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern int cudaresetdevice_bare(void);
