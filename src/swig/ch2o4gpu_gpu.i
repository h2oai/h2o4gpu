/* File : ch2o4gpu_gpu.i */
%module ch2o4gpu_gpu
%{
  #define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"

%init %{
    import_array();
%}

%include "elastic_net.i"
%include "kmeans.i"
%include "pca.i"
%include "pogs.i"
%include "tsvd.i"
%include "utils.i"
