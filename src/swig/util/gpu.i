/* File : gpu.i */
%{
extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern int get_gpu_info_c(int verbose,
                   int return_memory,
                   int return_name,
                   int return_usage,
                   int return_free_memory,
                   int return_capability,
                   int return_memory_by_pid,
                   int return_usage_by_pid,
                   int return_all,
 unsigned int *n_gpus, int *gpu_percent_usage,
 unsigned long long *gpu_total_memory, unsigned long long *gpu_free_memory,
 char **gpu_name,
 int *majors, int *minors,
 unsigned int *num_pids, unsigned int *pids, unsigned long long *usedGpuMemorys,
 unsigned int *num_pids_usage, unsigned int *pids_usage, unsigned long long *usedGpuUsage);
extern int cudaresetdevice_bare(void);
%}

%typemap(in) char** (char* tmp) {
    if ((SWIG_ConvertPtr($input, (void **) &tmp, $*1_descriptor, $disown | 0)) == -1) {
        tmp = NULL;
    }
    $1 = &tmp;
}

%typemap(argout) (char **strsplit) {
   int ntokens;
   PyObject *py_string_tmp;
   int py_err;

   for (ntokens = 0; $1[ntokens] != NULL; ntokens++) {
   }

   $result = PyList_New(ntokens);
   if (! $result) return NULL;

   for (int itoken = 0; itoken < ntokens; itoken++) {
       if ($1[itoken] == NULL) break;

       py_string_tmp = PyString_FromString( $1[itoken] );
       if (! py_string_tmp) return NULL;

       py_err = PyList_SetItem($result, itoken, py_string_tmp);
       if (py_err == -1) return NULL;
   }

   for (int itoken = 0; itoken < ntokens; itoken++) {
       free($1[itoken]);
   }
   free($1);

   return $result;
}

%apply int *OUTPUT {int *major, int *minor, int *ratioperf};

%apply int *OUTPUT {unsigned int *n_gpus};
%apply (int *INPLACE_ARRAY1) {int *gpu_percent_usage};
%apply (int *INPLACE_ARRAY1) {int *majors};
%apply (int *INPLACE_ARRAY1) {int *minors};
%apply (unsigned long long *INPLACE_ARRAY1) {unsigned long long *gpu_total_memory};
%apply (unsigned long long *INPLACE_ARRAY1) {unsigned long long *gpu_free_memory};
%apply (unsigned int *INPLACE_ARRAY1) {unsigned int *num_pids};
%apply (unsigned int *INPLACE_ARRAY1) {unsigned int *pids};
%apply (unsigned long long *INPLACE_ARRAY1) {unsigned long long *usedGpuMemorys};
%apply (unsigned int *INPLACE_ARRAY1) {unsigned int *num_pids_usage};
%apply (unsigned int *INPLACE_ARRAY1) {unsigned int *pids_usage};
%apply (unsigned long long *INPLACE_ARRAY1) {unsigned long long *usedGpuUsage};

extern int cudaresetdevice(int wDev, int nDev);
extern int get_compute_capability(int d_idx, int *major, int *minor, int *ratioperf);
extern int get_gpu_info_c(int verbose,
                   int return_memory,
                   int return_name,
                   int return_usage,
                   int return_free_memory,
                   int return_capability,
                   int return_memory_by_pid,
                   int return_usage_by_pid,
                   int return_all,
 unsigned int *n_gpus, int *gpu_percent_usage,
 unsigned long long *gpu_total_memory, unsigned long long *gpu_free_memory,
 char **gpu_name,
 int *majors, int *minors,
 unsigned int *num_pids, unsigned int *pids, unsigned long long *usedGpuMemorys,
 unsigned int *num_pids_usage, unsigned int *pids_usage, unsigned long long *usedGpuUsage);
extern int cudaresetdevice_bare(void);
