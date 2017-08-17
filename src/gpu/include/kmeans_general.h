#pragma once
#define MAX_NGPUS 16

#define CHECK 1

// TODO(pseudotensor): Avoid throw for python exception handling.  Need to avoid all exit's and return exit code all the way back.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__);
#define safe_cublas(ans) throw_on_cublas_error((ans), __FILE__, __LINE__);
#endif
