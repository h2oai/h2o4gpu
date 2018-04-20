/* File : gpu.i */
%{
extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern void get_gpu_info_c(unsigned int *n_gpus, int *gpu_percent_usage, int *gpu_total_memory, char **gpu_name);
extern int cudaresetdevice_bare(void);
%}

%apply int *OUTPUT {int *major, int *minor, int *ratioperf}
%apply int *OUTPUT {unsigned int *n_gpus}
%apply (int *INPLACE_ARRAY1) {int *gpu_percent_usage, int *gpu_total_memory, char **gpu_name};

extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern void get_gpu_info_c(unsigned int *n_gpus, int *gpu_percent_usage, int *gpu_total_memory, char **gpu_name);
extern int cudaresetdevice_bare(void);
