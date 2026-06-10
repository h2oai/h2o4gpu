/* File : ch2o4gpu_gpu.i */
%module ch2o4gpu_gpu
%{
  #define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"

%init %{
    import_array();
%}

%include "cpointer.i"
%include "solver/kmeans.i"
%include "solver/elastic_net.i"
%include "solver/pca.i"
%include "solver/pogs.i"
%include "solver/tsvd.i"
// factorization (ALS) solver disabled on CUDA 12: it uses cuSPARSE csrmm2,
// removed in CUDA 12, and is not consumed by DAI. Re-enable once the solver is
// ported to the generic cusparseSpMM API. See CMakeLists.txt GPU_SOURCES filter.
//%include "solver/factorization.i"
%include "solver/arima.i"
%include "matrix/matrix_dense.i"
%include "util/gpu.i"
