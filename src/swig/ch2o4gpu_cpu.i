/* File : ch2o4gpu_cpu.i */
%module ch2o4gpu_cpu
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
%include "solver/pogs.i"
%include "matrix/matrix_dense.i"
%include "metrics.i"