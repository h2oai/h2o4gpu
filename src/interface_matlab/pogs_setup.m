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
  eval(sprintf(['mex -largeArrayDims -I.. CFLAGS=''\\$CFLAGS -O3 %s'' ' ...
                'pogs_mex.cpp blas2cblas.cpp ../pogs.cpp -lmwblas ' ...
                '-output pogs'], omp_flag))
else
  unix('make clean -C ..');
  unix(sprintf(['export PATH=$PATH:%s;' ...
                'export DYLD_LIBRARY_PATH=%s:$DYLD_LIBRARY_PATH;' ...
                'make all -C .. IFLAGS="-D__MEX__"'], ...
               cuda_bin, cuda_lib));
  try
    eval(sprintf(['mex -largeArrayDims -I.. -D__CUDA__ -output pogs ' ...
                  'LDFLAGS=''\\$LDFLAGS -stdlib=libstdc++ -Wl,-rpath,%s'' ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -stdlib=libstdc++'' ' ...
                  'pogs_mex.cpp ../pogs_cu.o ../pogs_sp_cu.o ../pogs_cu_link.o ../cml.o ' ...
                  ' -L%s -lcudart -lcublas -lcusparse'], ...
                 cuda_lib, cuda_lib))
  catch
    fprintf('Linking to standard library failed, trying another.\n')
    eval(sprintf(['mex -largeArrayDims -I..  -D__CUDA__ -output pogs ' ...
                  'LDFLAGS=''\\$LDFLAGS -Wl,-rpath,%s'' ' ...
                  'pogs_mex.cpp ../pogs_cu.o ../pogs_sp_cu.o ../pogs_cu_link.o ../cml.o ' ...
                  '-L%s -lcudart -lcublas -lcusparse'], ...
                 cuda_lib, cuda_lib))
  end
end

savepath

fprintf('Setup Successful.\n')

