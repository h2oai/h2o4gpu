/* File : kmeans.i */
%{
extern int make_ptr_float_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                          const char ord, int k, int max_iterations, int init_from_data,
                          float threshold, const float *srcdata,
                          const float *centroids, float **pred_centroids, int **pred_labels);

extern int make_ptr_double_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                           const char ord, int k, int max_iterations, int init_from_data,
                           double threshold, const double *srcdata,
                           const double *centroids, double **pred_centroids, int **pred_labels);

extern int kmeans_transform_float(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const float *src_data, const float *centroids,
                           float **preds);

extern int kmeans_transform_double(int verbose,
                            int gpu_id, int n_gpu,
                            size_t m, size_t n, const char ord, int k,
                            const double *src_data, const double *centroids,
                            double **preds);
%}

%apply (float *IN_ARRAY1) {float *srcdata, float *centroids};
%apply (float **ARGOUT_ARRAY1) {float **pred_centroids, float **preds};

%apply (double *IN_ARRAY1) {double *srcdata, double *centroids};
%apply (double **ARGOUT_ARRAY1) {double **pred_centroids, double **preds};

%apply (int **ARGOUT_ARRAY1) {int **pred_labels};

extern int make_ptr_float_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                          const char ord, int k, int max_iterations, int init_from_data,
                          float threshold, const float *srcdata,
                          const float *centroids, float **pred_centroids, int **pred_labels);

extern int make_ptr_double_kmeans(int dopredict, int verbose, int seed, int gpu_id, int n_gpu, size_t mTrain, size_t n,
                           const char ord, int k, int max_iterations, int init_from_data,
                           double threshold, const double *srcdata,
                           const double *centroids, double **pred_centroids, int **pred_labels);

extern int kmeans_transform_float(int verbose,
                           int gpu_id, int n_gpu,
                           size_t m, size_t n, const char ord, int k,
                           const float *src_data, const float *centroids,
                           float **preds);

extern int kmeans_transform_double(int verbose,
                            int gpu_id, int n_gpu,
                            size_t m, size_t n, const char ord, int k,
                            const double *src_data, const double *centroids,
                            double **preds);