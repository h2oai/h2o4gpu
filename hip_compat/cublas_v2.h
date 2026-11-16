// cuBLAS names used by h2o4gpu, mapped to hipBLAS.
#pragma once

#include "cuda_runtime.h"
#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>

// Sources test the cuBLAS header guard to decide whether cuBLAS is available.
#ifndef CUBLAS_API_H_
#define CUBLAS_API_H_
#endif

#define cuFloatComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define cublasFillMode_t hipblasFillMode_t
#define cublasSideMode_t hipblasSideMode_t
#define cublasDiagType_t hipblasDiagType_t

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_UNIT HIPBLAS_DIAG_UNIT

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetStream hipblasGetStream

#define cublasSasum hipblasSasum
#define cublasDasum hipblasDasum
#define cublasSaxpy hipblasSaxpy
#define cublasDaxpy hipblasDaxpy
#define cublasCaxpy hipblasCaxpy
#define cublasZaxpy hipblasZaxpy
#define cublasSdgmm hipblasSdgmm
#define cublasDdgmm hipblasDdgmm
#define cublasSdot hipblasSdot
#define cublasDdot hipblasDdot
#define cublasSgeam hipblasSgeam
#define cublasDgeam hipblasDgeam
#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasSgemm_v2 hipblasSgemm
#define cublasDgemm_v2 hipblasDgemm
#define cublasSgemv hipblasSgemv
#define cublasDgemv hipblasDgemv
#define cublasSger hipblasSger
#define cublasDger hipblasDger
#define cublasIsamax hipblasIsamax
#define cublasIdamax hipblasIdamax
#define cublasSnrm2 hipblasSnrm2
#define cublasDnrm2 hipblasDnrm2
#define cublasSnrm2_v2 hipblasSnrm2
#define cublasDnrm2_v2 hipblasDnrm2
#define cublasScnrm2 hipblasScnrm2
#define cublasDznrm2 hipblasDznrm2
#define cublasSscal hipblasSscal
#define cublasDscal hipblasDscal
#define cublasSsyrk hipblasSsyrk
#define cublasDsyrk hipblasDsyrk
#define cublasStrsm hipblasStrsm
#define cublasDtrsm hipblasDtrsm
#define cublasStrsv hipblasStrsv
#define cublasDtrsv hipblasDtrsv
