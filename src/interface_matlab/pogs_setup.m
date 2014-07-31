function pogs_setup(varargin)
%% Setup script for pogs

% Parse input
opt_gpu = any(strcmp(varargin, '-gpu'));
opt_omp = any(strcmp(varargin, '-omp'));
opt_clb = any(strcmp(varargin, '-cuda_lib'));
opt_cbn = any(strcmp(varargin, '-cuda_bin'));

omp_flag = '';
if opt_omp
  omp_flag = '-fopenmp';
end

cuda_lib = '/usr/local/cuda/lib';
if opt_clb
  idx_lib = find(strcmp(varargin, '-cuda_lib')) + 1;
  cuda_lib = varargin{idx_lib};
end

cuda_bin = '/usr/local/cuda/bin';
if opt_cbn
  idx_bin = find(strcmp(varargin, '-cuda_bin')) + 1;
  cuda_bin = varargin{idx_bin};
end

if ~opt_gpu
  try
    unix(['make pogs.o -f Makefile -C .. IFLAGS="-fPIC -D__MEX__ ' omp_flag '"']);
    mex -v -largeArrayDims -I.. ...
        LDFLAGS='\$LDFLAGS' -lmwblas ...
        ../pogs.o pogs_mex.cpp blas2cblas.cpp ...
        -output pogs
  catch
    fprintf('Linking to standard library failed, trying another.\n')
    unix(['make pogs.o -f Makefile -C .. IFLAGS="-fPIC -D__MEX__ -stdlib=libstdc++ ' ...
          omp_flag '"']);
    mex -largeArrayDims -I.. ...
        LDFLAGS='\$LDFLAGS -stdlib=libstdc++' ...
        CXXFLAGS='\$CXXFLAGS -stdlib=libstdc++' ...
        ../pogs.o pogs_mex.cpp blas2cblas.cpp...
        -output pogs &> /dev/null

  end
else
  unix(sprintf(['export PATH=$PATH:%s;' ...
                'export DYLD_LIBRARY_PATH=%s:$DYLD_LIBRARY_PATH;' ...
                'make pogs_cu.o -f Makefile -C .. IFLAGS="-D__MEX__";' ...
                'make pogs_cu_link.o -f Makefile -C .. IFLAGS="-D__MEX__"'], ...
               cuda_bin, cuda_lib));
  try
    eval(sprintf(['mex -largeArrayDims -I.. -L%s -lcudart -lcublas -output pogs ' ...
                  'LDFLAGS=''\\$LDFLAGS -stdlib=libstdc++ -Wl,-rpath,%s'' ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -stdlib=libstdc++'' ' ...
                  'pogs_mex.cpp ../pogs_cu.o ../pogs_cu_link.o &> /dev/null'], ...
                 cuda_lib, cuda_lib))
  catch
    fprintf('Linking to standard library failed, trying another.\n')
    eval(sprintf(['mex -largeArrayDims -I.. -L%s -lcudart -lcublas -output pogs ' ...
                  'LDFLAGS=''\\$LDFLAGS -Wl,-rpath,%s'' ' ...
                  'pogs_mex.cpp ../pogs_cu.o ../pogs_cu_link.o'], ...
                 cuda_lib, cuda_lib))
  end
end

savepath

fprintf('Setup Successful.\n')

