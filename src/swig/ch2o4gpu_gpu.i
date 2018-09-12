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
%include "solver/ffm.i"
%include "matrix/matrix_dense.i"
%include "util/gpu.i"
