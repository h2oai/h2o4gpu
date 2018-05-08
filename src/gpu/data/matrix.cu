#include "matrix.cuh"
#include <algorithm>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

namespace matrix
{
	using namespace h2o4gpu;

	void max_index_per_column(Matrix<float>& A, std::vector<int>& result_array, device::DeviceContext& context){

		int result;
		for (int i=0; i<A.columns(); i++) {
			safe_cublas(cublasIsamax(context.cublas_handle, A.rows(), A.data() + i*A.rows(), 1, &result));
			result_array[i] = result - 1 + i * A.rows();
		}
	}

	void max_index_per_column(Matrix<double>& A, std::vector<int>& result_array, device::DeviceContext& context){

		int result;
		for (int i=0; i<A.columns(); i++) {
			safe_cublas(cublasIdamax(context.cublas_handle, A.rows(), A.data() + i*A.rows(), 1, &result));
			result_array[i] = result - 1 + i * A.rows();
		}
	}

	template<typename T, typename U>
	void multiply(Matrix<T>& A, const U a, device::DeviceContext& context)
	{
		thrust::transform(A.dptr(), A.dptr() + A.size(), A.dptr(), [=]__device__ (U val)
						  {
							  return val * a;
						  }
		);
	}

	template<typename T>
	void subtract(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, device::DeviceContext& context)
	{
		auto counting = thrust::make_counting_iterator(0);
		const T* d_A = A.data();
		const T* d_B = B.data();
		T* d_C = C.data();
		thrust::for_each(counting, counting + A.rows() * A.columns(), [=]__device__(int idx)
						 {
							 d_C[idx] = d_A[idx] - d_B[idx];
						 });
	}

	template<typename T>
	void add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, device::DeviceContext& context)
	{
		auto counting = thrust::make_counting_iterator(0);
		const T* d_A = A.data();
		const T* d_B = B.data();
		T* d_C = C.data();
		thrust::for_each(counting, counting + A.rows() * A.columns(), [=]__device__(int idx)
						 {
							 d_C[idx] = d_A[idx] + d_B[idx];
						 });
	}


	template<typename T>
	void normalize_vector_thrust(Matrix<T>& M, device::DeviceContext& context){
		float M_inner = thrust::inner_product(M.dptr(), M.dptr() + M.size(), M.dptr(), 0.0f); //Will allocate memory for every call to fxn.
		M.transform([=]__device__ (float val){return val / std::sqrt(M_inner);});
	}

	void multiply_diag(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context, bool left_diag)
	{
		cublasSideMode_t mode = left_diag ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

		int m = C.rows();
		int n = C.columns();
		int lda = m;
		int incx = 1; //Review what this should be...
		int ldc = m;

		safe_cublas(cublasSdgmm(context.cublas_handle, mode, m, n, A.data(), lda, B.data(), incx, C.data(), ldc));
	}

	void multiply_diag(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context, bool left_diag)
	{
		cublasSideMode_t mode = left_diag ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

		int m = C.rows();
		int n = C.columns();
		int lda = m;
		int incx = 1; //Review what this should be...
		int ldc = m;

		safe_cublas(cublasDdgmm(context.cublas_handle, mode, m, n, A.data(), lda, B.data(), incx, C.data(), ldc));
	}

	void multiply(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context, bool transpose_a, bool transpose_b, float alpha)
	{
		cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		const float beta = 0;

		int m = C.rows();
		int n = C.columns();
		int k = transpose_a ? A.rows() : A.columns();
		int lda = transpose_a ? k : m;
		int ldb = transpose_b ? n : k;
		int ldc = m;

		safe_cublas(cublasSgemm(context.cublas_handle, op_a, op_b, m, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
	}

	void multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context, bool transpose_a, bool transpose_b, double alpha)
	{
		cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		const double beta = 0;

		int m = C.rows();
		int n = C.columns();
		int k = transpose_a ? A.rows() : A.columns();
		int lda = transpose_a ? k : m;
		int ldb = transpose_b ? n : k;
		int ldc = m;

		safe_cublas(cublasDgemm(context.cublas_handle, op_a, op_b, m, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
	}

	void transpose(const Matrix<float>& A, Matrix<float>& B, device::DeviceContext& context)
	{
		h2o4gpu_check(A.rows() == B.columns()&&A.columns() == B.rows(), "Transpose dimensions incorrect");
		const float alpha = 1.0f;
		const float beta = 0.0f;
		safe_cublas(cublasSgeam(context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, B.rows(), B.columns(), &alpha, A.data(), A.rows(), &beta, NULL, B.rows(), B.data(), B.rows()));
	}

	void transpose(const Matrix<double>& A, Matrix<double>& B, device::DeviceContext& context)
	{
		h2o4gpu_check(A.rows() == B.columns()&&A.columns() == B.rows(), "Transpose dimensions incorrect");
		const double alpha = 1.0f;
		const double beta = 0.0f;
		safe_cublas(cublasDgeam(context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, B.rows(), B.columns(), &alpha, A.data(), A.rows(), &beta, NULL, B.rows(), B.data(), B.rows()));
	}

	void normalize_columns(Matrix<float>& M, Matrix<float>& M_temp, Matrix<float>& column_length, const Matrix<float>& ones, device::DeviceContext& context)
	{
		thrust::transform(M.dptr(), M.dptr() + M.size(), M_temp.dptr(), sqr_op());
		auto d_column_length = column_length.data();
		auto d_ones = ones.data();
		const float alpha = 1.0f;
		const float beta = 0.0f;
		safe_cublas(cublasSgemv(context.cublas_handle, CUBLAS_OP_T, M.rows(), M.columns(), &alpha, M_temp.data(), M.rows(), d_ones, 1, &beta, d_column_length, 1));

		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(float val)
		                  {
							  if (val == 0.0)
							  {
								  return 0.0;
							  }

			                  return 1.0/ sqrt(val);
		                  });

		safe_cublas(cublasSdgmm(context.cublas_handle, CUBLAS_SIDE_RIGHT, M.rows(), M.columns(), M.data(), M.rows(), d_column_length, 1, M.data(), M.rows()));
	}

	void normalize_columns(Matrix<double>& M, Matrix<double>& M_temp, Matrix<double>& column_length, const Matrix<double>& ones, device::DeviceContext& context)
	{
		thrust::transform(M.dptr(), M.dptr() + M.size(), M_temp.dptr(), sqr_op());
		auto d_column_length = column_length.data();
		auto d_ones = ones.data();
		const double alpha = 1.0f;
		const double beta = 0.0f;
		safe_cublas(cublasDgemv(context.cublas_handle, CUBLAS_OP_T, M.rows(), M.columns(), &alpha, M_temp.data(), M.rows(), d_ones, 1, &beta, d_column_length, 1));

		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(double val)
		                  {
							  if (val == 0.0)
							  {
								  return 0.0;
							  }

			                  return 1.0/ sqrt(val);
		                  });

		safe_cublas(cublasDdgmm(context.cublas_handle, CUBLAS_SIDE_RIGHT, M.rows(), M.columns(), M.data(), M.rows(), d_column_length, 1, M.data(), M.rows()));
	}

	void normalize_columns(Matrix<float>& M, device::DeviceContext& context)
	{
		Matrix<float> M_temp(M.rows(), M.columns());
		Matrix<float> columns_length(1, M.columns());
		Matrix<float> ones(1, M.columns());
		ones.fill(1.0f);
		normalize_columns(M, M_temp, columns_length, ones, context);
	}

	void normalize_columns(Matrix<double>& M, device::DeviceContext& context)
	{
		Matrix<double> M_temp(M.rows(), M.columns());
		Matrix<double> columns_length(1, M.columns());
		Matrix<double> ones(1, M.columns());
		ones.fill(1.0f);
		normalize_columns(M, M_temp, columns_length, ones, context);
	}

	void normalize_vector_cublas(Matrix<float>& M, device::DeviceContext& context){
        float norm2 = 0.0;
        safe_cublas(cublasSnrm2(context.cublas_handle, M.rows(), M.data(), 1.0, &norm2));
        M.transform([=]__device__ (float val){return val * (1/norm2);});
    }

	void normalize_vector_cublas(Matrix<double>& M, device::DeviceContext& context){
        double norm2 = 0.0;
        safe_cublas(cublasDnrm2(context.cublas_handle, M.rows(), M.data(), 1.0, &norm2));
        M.transform([=]__device__ (float val){return val * (1/norm2);});
    }

	void residual(const Matrix<float>& X, const Matrix<float>& D, const Matrix<float>& S, Matrix<float>& R, device::DeviceContext& context)
	{
		multiply(D, S, R, context);
		subtract(X, R, R, context);
	}

	void residual(const Matrix<double>& X, const Matrix<double>& D, const Matrix<double>& S, Matrix<double>& R, device::DeviceContext& context)
	{
		multiply(D, S, R, context);
		subtract(X, R, R, context);
	}

	void calculate_eigen_pairs_exact(const Matrix<float>& X, Matrix<float>& Q, Matrix<float>& w, device::DeviceContext& context)
	{
		h2o4gpu_check(X.rows() == X.columns(), "X must be a symmetric matrix");
		h2o4gpu_check(X.rows() == Q.rows() && X.columns() == Q.columns(), "X and Q must have the same dimension");
		h2o4gpu_check(w.rows() == Q.columns(), "Q and w should have the same number of columns");

		int lwork;
		safe_cusolver(cusolverDnSsyevd_bufferSize(context.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, X.rows(), X.data(), X.columns(), w.data(), &lwork));

		float *d_work;
		safe_cuda(cudaMalloc(&d_work, sizeof(float) * lwork));

		int *dev_info = NULL;
		safe_cuda(cudaMalloc ((void**)&dev_info, sizeof(int)));
		Q.copy(X);
		safe_cusolver(cusolverDnSsyevd(context.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, Q.rows(), Q.data(), Q.columns(), w.data(), d_work, lwork, dev_info));
		safe_cuda(cudaDeviceSynchronize());
		safe_cuda(cudaFree(d_work));
		safe_cuda(cudaFree(dev_info));
		safe_cuda(cudaGetLastError());
	}

	void calculate_eigen_pairs_exact(const Matrix<double>& X, Matrix<double>& Q, Matrix<double>& w, device::DeviceContext& context)
	{
		h2o4gpu_check(X.rows() == X.columns(), "X must be a symmetric matrix");
		h2o4gpu_check(X.rows() == Q.rows() && X.columns() == Q.columns(), "X and Q must have the same dimension");
		h2o4gpu_check(w.rows() == Q.columns(), "Q and w should have the same number of columns");

		int lwork;
		safe_cusolver(cusolverDnDsyevd_bufferSize(context.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, X.rows(), X.data(), X.columns(), w.data(), &lwork));

		double *d_work;
		safe_cuda(cudaMalloc(&d_work, sizeof(double) * lwork));

		int *dev_info = NULL;
		safe_cuda(cudaMalloc ((void**)&dev_info, sizeof(int)));
		Q.copy(X);
		safe_cusolver(cusolverDnDsyevd(context.cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, Q.rows(), Q.data(), Q.columns(), w.data(), d_work, lwork, dev_info));
		safe_cuda(cudaDeviceSynchronize());
		safe_cuda(cudaFree(d_work));
		safe_cuda(cudaFree(dev_info));
		safe_cuda(cudaGetLastError());
	}

	void dot_product(Matrix<float>& b_k1, Matrix<float>& b_k, float* eigen_value_estimate, device::DeviceContext& context)
	{
		safe_cublas(cublasSdot(context.cublas_handle, b_k1.rows(), b_k1.data(), 1.0, b_k.data(), 1.0, eigen_value_estimate));
	}

	void dot_product(Matrix<double>& b_k1, Matrix<double>& b_k, double* eigen_value_estimate, device::DeviceContext& context)
	{
		safe_cublas(cublasDdot(context.cublas_handle, b_k1.rows(), b_k1.data(), 1.0, b_k.data(), 1.0, eigen_value_estimate));
	}

	//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Stricly floating point operations that are not used
	void linear_solve(const Matrix<float>& A, Matrix<float>& X, const Matrix<float>& B, device::DeviceContext& context)
	{
		h2o4gpu_check(A.rows()>= A.columns(),"Linear solve requires m >= n");
		h2o4gpu_check(X.rows()>= X.columns(),"Linear solve requires n >= k"); //TODO: is this restriction necessary?

		Matrix<float> A_copy(A);
		Matrix<float> B_copy(A.rows(), A.columns());
		thrust::copy(B.dptr(), B.dptr() + B.size(), B_copy.dptr());
		thrust::fill(B_copy.dptr() + B.size(), B_copy.dptr() + B_copy.size(), 0.0f);

		int work_size = 0;
		safe_cusolver(cusolverDnSgeqrf_bufferSize(context.cusolver_handle, A_copy.rows(), A_copy.columns(), A_copy.data(), A_copy.rows(), &work_size));

		thrust::device_vector<float> work(work_size);
		float* d_work = thrust::raw_pointer_cast(work.data());

		thrust::device_vector<float> tau((std::min)(A.rows(), A.columns()));
		float* d_tau = thrust::raw_pointer_cast(tau.data());

		thrust::device_vector<int> dev_info(1);
		int* d_dev_info = thrust::raw_pointer_cast(dev_info.data());

		safe_cusolver(cusolverDnSgeqrf(context.cusolver_handle, A_copy.rows(), A_copy.columns(), A_copy.data(), A_copy.rows(), d_tau, d_work, work_size, d_dev_info));

		h2o4gpu_check(dev_info[0] == 0, "geqrf unsuccessful");

		safe_cusolver(cusolverDnSormqr(context.cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, A.rows(), A.columns(), (std::min)(A.rows(), A.columns()), A_copy.data(), A.rows(), d_tau, B_copy.data(), A.rows(), d_work, work_size, d_dev_info));
		h2o4gpu_check(dev_info[0] == 0, "ormqr unsuccessful");

		Matrix<float> R(A.columns(), A.columns());
		Matrix<float> QTB(A.columns(), B.columns());
		auto counting = thrust::make_counting_iterator(0);
		int n = R.columns();
		int m = A.rows();
		auto d_R = R.data();
		auto d_A_copy = A_copy.data();
		auto d_QTB = QTB.data();
		auto d_B_copy = B_copy.data();
		int qtb_columns = QTB.columns();
		thrust::for_each(counting, counting + R.size(), [=]__device__ (int idx)
		                 {
			                 int row = idx % n;
			                 int column = idx / n;
			                 d_R[idx] = d_A_copy[column * m + row];

			                 if (column < qtb_columns)
			                 {
				                 d_QTB[idx] = d_B_copy[column * m + row];
			                 }
		                 });

		const float alpha = 1.0f;
		safe_cublas(cublasStrsm(context.cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, QTB.rows(), QTB.columns(), &alpha, R.data(), R.rows(), QTB.data(), QTB.rows()));

		thrust::copy(QTB.dptr(), QTB.dptr() + QTB.size(), X.data());
	}

	void pseudoinverse(const Matrix<float>& A, Matrix<float>& pinvA, device::DeviceContext& context)
	{
		h2o4gpu_check(A.rows() == pinvA.columns() && A.columns() == pinvA.rows(), "pseudoinverse dimensions incorrect");

		//Add zero rows if m < n such that m >= n
		Matrix<float> A_extended((std::max)(A.columns(), A.rows()), A.columns());
		auto counting = thrust::make_counting_iterator(0);
		int A_column_size = A.rows();
		int A_extended_column_size = A_extended.rows();
		auto d_A = A.data();
		auto d_A_extended = A_extended.data();

		thrust::for_each(counting, counting + A_extended.size(), [=]__device__(int idx)
		                 {
			                 int row = idx % A_extended_column_size;

			                 if (row < A_column_size)
			                 {
				                 int column = idx / A_extended_column_size;
				                 d_A_extended[idx] = d_A[A_column_size * column + row];
			                 }
			                 else
			                 {
				                 d_A_extended[idx] = 0;
			                 }
		                 });

		int work_size = 0;
		safe_cusolver(cusolverDnSgesvd_bufferSize(context.cusolver_handle, A_extended.rows(), A_extended.columns(), &work_size));

		Matrix<float> work(work_size, 1);

		Matrix<float> S((std::min)(A_extended.rows(), A_extended.columns()), 1);
		Matrix<float> U(A_extended.rows(), A_extended.rows());
		Matrix<float> VT(A_extended.columns(), A_extended.columns());
		Matrix<int> dev_info(1, 1);

		safe_cusolver (cusolverDnSgesvd(context.cusolver_handle, 'A', 'A', A_extended.rows(), A_extended.columns(), d_A_extended, A_extended.rows(), S.data(), U.data(), U.rows(), VT.data(), VT.rows(), work.data(), work_size, NULL, dev_info.data()));

		float eps = 1e-5;
		thrust::transform(S.dptr(), S.dptr() + S.size(), S.dptr(), [=]__device__(float val)
		                  {
			                  if (abs(val) < eps)
			                  {
				                  return 0.0;
			                  }
			                  else
			                  {
				                  return 1.0 / val;
			                  }
		                  });

		Matrix<float> UT(A_extended.rows(), A_extended.rows());

		//Calculate transpose of U
		const float alpha = 1.0;
		const float beta = 0.0;
		safe_cublas(cublasSgeam(context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, UT.rows(), UT.columns(), &alpha, U.data(), UT.rows(), &beta,NULL, UT.rows(), UT.data(), UT.rows()));

		safe_cublas(cublasSdgmm(context.cublas_handle, CUBLAS_SIDE_LEFT, UT.rows(), UT.columns(), UT.data(), UT.rows(), S.data(), 1, U.data(), U.rows()));

		Matrix<float> pinvA_extended(A_extended.columns(), A_extended.rows());
		multiply(VT, U, pinvA_extended, context, true);

		thrust::copy(pinvA_extended.dptr(), pinvA_extended.dptr() + pinvA.size(), pinvA.dptr());
	}

	void f_normalize(Matrix<float>& M, device::DeviceContext& context)
	{
		Matrix<float> temp(M.rows(), M.columns());
		thrust::transform(M.dptr(), M.dptr() + M.size(), temp.dptr(), sqr_op());
		float sum = thrust::reduce(temp.dptr(), temp.dptr() + temp.size());
		multiply(M, 1.0 / std::sqrt(sum), context);
		thrust::transform(M.dptr(), M.dptr() + M.size(), temp.dptr(), sqr_op());
		float final_sum = thrust::reduce(temp.dptr(), temp.dptr() + temp.size());
		printf("f norm sum squares: %1.4f\n", final_sum);
	}

	void normalize_columns_cub(Matrix<float>& M, device::DeviceContext& context)
	{
		//Create alias so device Lamba does not dereference this pointer
		int m = M.rows();

		thrust::device_vector<float> temp(M.size());
		thrust::device_vector<float> length_squared(M.columns());

		thrust::transform(M.dptr(), M.dptr() + M.size(), temp.begin(), [=]__device__(float val)
		                  {
			                  return val * val;
		                  });


		thrust::device_vector<int> column_segments(M.columns() + 1);
		auto counting = thrust::make_counting_iterator(0);
		thrust::transform(counting, counting + column_segments.size(), column_segments.begin(), [=]__device__(int idx)
		                  {
			                  return idx * m;
		                  });

		// Determine temporary device storage requirements
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		auto segments = thrust::raw_pointer_cast(column_segments.data());
		auto sum_in = thrust::raw_pointer_cast(temp.data());
		auto sum_out = thrust::raw_pointer_cast(length_squared.data());
		cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, sum_in, sum_out,
		                                M.columns(), segments, segments + 1);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, sum_in, sum_out,
		                                M.columns(), segments, segments + 1);

		//Scale
		auto d_length_squared = thrust::raw_pointer_cast(length_squared.data());
		auto d_data = M.data();
		thrust::transform(counting, counting + M.size(), M.dptr(), [=]__device__(int idx)
		                  {
			                  int col = idx / m;

			                  float length_squared = d_length_squared[col];

			                  if (length_squared > 0.0)
			                  {
				                  return d_data[idx] / std::sqrt(d_length_squared[col]);
			                  }
			                  else
			                  {
				                  return 0.0f;
			                  }
		                  });

		cudaFree(d_temp_storage);
	}

}

//Orignal Impl
template void matrix::multiply<double>(Matrix<double>& A, const float a, device::DeviceContext& context);

//Impl for floats and doubles
template void matrix::multiply<float>(Matrix<float>& A, const float a, device::DeviceContext& context);
template void matrix::multiply<double>(Matrix<double>& A, const double a, device::DeviceContext& context);
template void matrix::subtract<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context);
template void matrix::subtract<double>(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context);
template void matrix::add<float>(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context);
template void matrix::add<double>(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context);
template void matrix::normalize_vector_thrust<float>(Matrix<float>& M, device::DeviceContext& context);
template void matrix::normalize_vector_thrust<double>(Matrix<double>& M, device::DeviceContext& context);

