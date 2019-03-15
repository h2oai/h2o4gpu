#ifndef SRC_INCLUDE_SOLVER_FACTORIZATION_H

void free_data_float(float **ptr);

void free_data_double(double **ptr);

void free_data_int(int **ptr);

void copy_fecatorization_result_float(float *dst, const float **src,
                                      const int size);

void copy_fecatorization_result_double(double *dst, const double **src,
                                       const int size);

int make_factorization_data_double(
    const int m, const int n, const int f, const long nnz, const long nnz_test,
    const int *csrRowIndexHostPtr, const int *csrColIndexHostPtr,
    const double *csrValHostPtr, const int *cscRowIndexHostPtr,
    const int *cscColIndexHostPtr, const double *cscValHostPtr,
    const int *cooRowIndexHostPtr, const int *cooColIndexHostPtr,
    const double *cooValHostPtr, double *thetaTHost, double *XTHost,
    const int *cooRowIndexTestHostPtr, const int *cooColIndexTestHostPtr,
    const double *cooValTestHostPtr,

    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    double **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, double **cscValDevicePtr,
    int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr,
    double **cooValDevicePtr, double **thetaTDevice, double **XTDevice,
    int **cooRowIndexTestDevicePtr, int **cooColIndexTestDevicePtr,
    double **cooValTestDevicePtr);

int make_factorization_data_float(
    const int m, const int n, const int f, const long nnz, const long nnz_test,
    const int *csrRowIndexHostPtr, const int *csrColIndexHostPtr,
    const float *csrValHostPtr, const int *cscRowIndexHostPtr,
    const int *cscColIndexHostPtr, const float *cscValHostPtr,
    const int *cooRowIndexHostPtr, const int *cooColIndexHostPtr,
    const float *cooValHostPtr, float *thetaTHost, float *XTHost,
    const int *cooRowIndexTestHostPtr, const int *cooColIndexTestHostPtr,
    const float *cooValTestHostPtr,

    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    float **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, float **cscValDevicePtr,
    int **cooRowIndexDevicePtr, int **cooColIndexDevicePtr,
    float **cooValDevicePtr, float **thetaTDevice, float **XTDevice,
    int **cooRowIndexTestDevicePtr, int **cooColIndexTestDevicePtr,
    float **cooValTestDevicePtr);

int run_factorization_step_double(
    const int m, const int n, const int f, const long nnz, const double lambda,
    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    double **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, double **cscValDevicePtr, double **thetaTDevice,
    double **XTDevice, const int X_BATCH, const int THETA_BATCH);

int run_factorization_step_float(
    const int m, const int n, const int f, const long nnz, const float lambda,
    int **csrRowIndexDevicePtr, int **csrColIndexDevicePtr,
    float **csrValDevicePtr, int **cscRowIndexDevicePtr,
    int **cscColIndexDevicePtr, float **cscValDevicePtr, float **thetaTDevice,
    float **XTDevice, const int X_BATCH, const int THETA_BATCH);

float factorization_score_float(const int m, const int n, const int f,
                                const long nnz, const float lambda,
                                float **thetaTDevice, float **XTDevice,
                                int **cooRowIndexDevicePtr,
                                int **cooColIndexDevicePtr,
                                float **cooValDevicePtr);

double factorization_score_double(const int m, const int n, const int f,
                                  const long nnz, const float lambda,
                                  double **thetaTDevice, double **XTDevice,
                                  int **cooRowIndexDevicePtr,
                                  int **cooColIndexDevicePtr,
                                  double **cooValDevicePtr);

#endif