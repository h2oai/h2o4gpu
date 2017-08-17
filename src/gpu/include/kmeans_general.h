#pragma once
#define MAX_NGPUS 16

#define CHECK 1

// TODO(pseudotensor): Avoid throw for python exception handling.  Need to avoid all exit's and return exit code all the way back.
#define gpuErrchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__);
#define safe_cublas(ans) throw_on_cublas_error((ans), __FILE__, __LINE__);

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

inline cudaError_t throw_on_cuda_error(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }

    return code;
}

// TODO move to kmeans_general
inline cublasStatus_t throw_on_cublas_error(cublasStatus_t code, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error: %s %s %d\n", cudaGetErrorEnum(code), file, line);
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }

    return code;
}