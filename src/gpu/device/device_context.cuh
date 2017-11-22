#pragma once
#include "cublas_v2.h"
#include "../tsvd/utils.cuh"
#include <cusparse.h>
#include <cusolverDn.h>

namespace tsvd
{
	class DeviceContext
	{
	public:
		cublasHandle_t cublas_handle;
		cusolverDnHandle_t cusolver_handle;
		cusparseHandle_t cusparse_handle;
		CubMemory cub_mem;

		DeviceContext()
		{
			safe_cublas(cublasCreate(&cublas_handle));
			safe_cusolver(cusolverDnCreate(&cusolver_handle));
			safe_cusparse(cusparseCreate(&cusparse_handle));
		}

		~DeviceContext()
		{
			safe_cublas(cublasDestroy(cublas_handle));
			safe_cusolver(cusolverDnDestroy(cusolver_handle));
			safe_cusparse(cusparseDestroy(cusparse_handle));
		}
	};
}
