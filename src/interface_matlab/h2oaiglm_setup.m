function h2oaiglm_setup(varargin)
%% Setup script for h2oaiglm

% Parse input
opt_gpu = any(strcmp(varargin, '-gpu'));
opt_omp = any(strcmp(varargin, '-omp'));
opt_clb = any(strcmp(varargin, '-cuda_lib'));
opt_cbn = any(strcmp(varargin, '-cuda_bin'));

omp_flag = '';
if opt_omp
  omp_flag = '-fopenmp';
end

cuda_lib = '/usr/local/cuda/lib64';
if ismac
  cuda_lib = '/usr/local/cuda/lib';
end
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
  unix('make clean -C ..');
  unix(sprintf('make cpu -C .. IFLAGS="-D__MEX__ %s"', omp_flag));
  eval(sprintf(['mex -largeArrayDims -I../include ' ...
                'CFLAGS=''\\$CFLAGS -O3 %s'' ' ...
                'CXXFLAGS=''\\$CXXFLAGS -std=c++11'' ' ...
                'LDFLAGS=''\\$LDFLAGS %s'' ' ...
                'h2oaiglm_mex.cpp blas2cblas.cpp ../build/h2oaiglm.a -lmwblas  ' ...
                '-output h2oaiglm'], omp_flag, omp_flag))
else
  unix('make clean -C ..');
  unix(sprintf(['export PATH=$PATH:%s;' ...
                'export DYLD_LIBRARY_PATH=%s:$DYLD_LIBRARY_PATH;' ...
                'make gpu -C .. IFLAGS="-D__MEX__"'], ...
               cuda_bin, cuda_lib));
  try
    eval(sprintf(['mex -largeArrayDims -I../include -I../gpu/include ' ...
                  '-D__CUDA__ -output h2oaiglm ' ...
                  'LDFLAGS=''\\$LDFLAGS -stdlib=libstdc++ -Wl,-rpath,%s'' ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -stdlib=libstdc++ -std=c++11'' ' ...
                  'h2oaiglm_mex.cpp ../build/h2oaiglm.a ' ...
                  ' -L%s -lcudart -lcublas -lcusparse'], ...
                 cuda_lib, cuda_lib))
  catch
    fprintf('Linking to standard library failed, trying another.\n')
    eval(sprintf(['mex -largeArrayDims -I../include -I../gpu/include ' ...
                  '-D__CUDA__ -output h2oaiglm ' ...
                  'LDFLAGS=''\\$LDFLAGS -Wl,-rpath,%s'' ' ...
                  'CXXFLAGS=''\\$CXXFLAGS -std=c++11'' ' ...
                  'h2oaiglm_mex.cpp ../build/h2oaiglm.a ' ...
                  '-L%s -lcudart -lcublas -lcusparse'], ...
                 cuda_lib, cuda_lib))
  end
end

savepath

fprintf('Setup Successful.\n')

