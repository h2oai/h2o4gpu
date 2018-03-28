/* File : ch2o4gpu_cpu.i */
%module ch2o4gpu_cpu
%{
  #define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"

%init %{
    import_array();
%}

%include "elastic_net.i"
%include "kmeans.i"
%include "pogs.i"
