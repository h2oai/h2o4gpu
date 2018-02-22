#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "../device/device_context.cuh"
#include "tsvd.h"
#include <ctime>
#include <thrust/iterator/counting_iterator.h>
#include<algorithm>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

namespace tsvd
{

	/**
	 * Division utility to get explained variance ratio
	 *
	 * @param XVar
	 * @param XVarSum
	 * @param ExplainedVarRatio
	 * @param context
	 */
	template<typename T>
	void divide(const Matrix<T> &XVar, const Matrix<T> &XVarSum, Matrix<T> &ExplainedVarRatio, DeviceContext &context){
		auto d_x_var = XVar.data();
		auto d_x_var_sum = XVarSum.data();
		auto d_expl_var_ratio = ExplainedVarRatio.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+ExplainedVarRatio.size(), [=]__device__(int idx){
			float div_val = 0.0;
			//XVarSum can possibly be zero
			if(d_x_var_sum[0] != 0.0){
				div_val = d_x_var[idx] / d_x_var_sum[0];
			}
			d_expl_var_ratio[idx] = div_val;
		} );
	}

	/**
	 * Square each value in a matrix
	 *
	 * @param UmultSigma
	 * @param UmultSigmaSquare
	 * @param context
	 */
	template<typename T>
	void square_val(const Matrix<T> &UmultSigma, Matrix<T> &UmultSigmaSquare, DeviceContext &context){
		auto n = UmultSigma.columns();
		auto m = UmultSigma.rows();
		auto k = UmultSigmaSquare.rows();
		auto d_u_mult_sigma = UmultSigma.data();
		auto d_u_mult_sigma_square = UmultSigmaSquare.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+UmultSigmaSquare.size(), [=]__device__(int idx){
			float square_val = std::pow(d_u_mult_sigma[idx],2);
			d_u_mult_sigma_square[idx] = square_val;
		} );
	}

	/**
	 * Alternative variance calculation (Can be slow for big matrices)
	 *
	 * @param UmultSigma
	 * @param k
	 * @param UmultSigmaVar
	 * @param context
	 */
	template<typename T>
	void calc_var(const Matrix<T>UmultSigma, int k, Matrix<T> &UmultSigmaVar, DeviceContext &context){
		//Set aside matrix of 1's for getting columnar sums(t(UmultSima) * UmultOnes)
		Matrix<T>UmultOnes(UmultSigma.rows(), 1);
		UmultOnes.fill(1.0f);

		//Allocate matrices for variance calculation
		Matrix<T>UmultSigmaSquare(UmultSigma.rows(), UmultSigma.columns());
		Matrix<T>UmultSigmaSum(k, 1);
		Matrix<T>UmultSigmaSumSquare(k, 1);
		Matrix<T>UmultSigmaSumOfSquare(k, 1);
		Matrix<T>UmultSigmaVarNum(k, 1);

		//Calculate Variance
		square_val(UmultSigma, UmultSigmaSquare, context);
		multiply(UmultSigmaSquare, UmultOnes, UmultSigmaSumOfSquare, context, true, false, 1.0f);
		multiply(UmultSigma, UmultOnes, UmultSigmaSum, context, true, false, 1.0f);
		square_val(UmultSigmaSum, UmultSigmaSumSquare, context);
		//Get rows
		auto m = UmultSigma.rows();
		multiply(UmultSigmaSumOfSquare, m, context);
		subtract(UmultSigmaSumOfSquare, UmultSigmaSumSquare, UmultSigmaVarNum, context);
		auto d_u_sigma_var_num = UmultSigmaVarNum.data();
		auto d_u_sigma_var = UmultSigmaVar.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+UmultSigmaVar.size(), [=]__device__(int idx){
			float div_val = d_u_sigma_var_num[idx]/(std::pow(m,2));
			d_u_sigma_var[idx] = div_val;
		} );
	}


	template<typename T>
	class variance_iterator{
	public:
		// Required iterator traits
		typedef variance_iterator<T>          self_type;            ///< My own type
		typedef size_t                            difference_type;  ///< Type to express the result of subtracting one iterator from another
		typedef T                           value_type;             ///< The type of the element the iterator can point to
		typedef T*                          pointer;                ///< The type of a pointer to an element the iterator can point to
		typedef T                           reference;              ///< The type of a reference to an element the iterator can point to
		typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
		const T* data_ptr;
		const T* mean_ptr;
		const int col_rows;
		size_t offset;
		__device__ T operator[](size_t idx){
			idx = idx + offset;
			T mean = mean_ptr[idx/col_rows];
			T dev_square = pow((data_ptr[idx] - mean),2);
			return dev_square;
		}
		__device__ self_type operator+(size_t idx){
			self_type retval(data_ptr, mean_ptr, col_rows);
			retval.offset += idx;
			return retval;
		}

		__host__ __device__ variance_iterator(const T* data_ptr, const T* mean_ptr, const int col_rows):data_ptr(data_ptr), mean_ptr(mean_ptr), col_rows(col_rows), offset(0){

		}
	};

	/**
	 * Utility to calculate variance for each column of a matrix
	 *
	 * @param X
	 * @param UColMean
	 * @param UVar
	 * @param context
	 */
	template<typename T>
	void calc_var_numerator(const Matrix<T> &X, const Matrix<T> &UColMean, Matrix<T> &UVar, DeviceContext &context){
		auto m = X.rows();
		variance_iterator<T> variance(X.data(), UColMean.data(), m);
		thrust::device_vector<int> segments(X.columns() + 1);
		thrust::sequence(segments.begin(), segments.end(), 0, static_cast<int>(X.rows()));
		// Determine temporary device storage requirements
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		int cols = static_cast<int>(X.columns());
		cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, variance, UVar.data(),
										cols, thrust::raw_pointer_cast(segments.data()), thrust::raw_pointer_cast(segments.data() + 1));
		// Allocate temporary storage
		safe_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run sum-reduction
		cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, variance, UVar.data(),
										cols, thrust::raw_pointer_cast(segments.data()), thrust::raw_pointer_cast(segments.data() + 1));
		safe_cuda(cudaFree(d_temp_storage));

	}

	/**
	 * Utility to reverse q to show most import k to least important k
	 *
	 * @param Q
	 * @param QReversed
	 * @param context
	 */
	template<typename T>
	void col_reverse_q(const Matrix<T> &Q, Matrix<T> &QReversed, DeviceContext &context){
		auto n = Q.columns();
		auto m = Q.rows();
		auto k = QReversed.rows();
		auto d_q = Q.data();
		auto d_q_reversed = QReversed.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+QReversed.size(), [=]__device__(int idx){
			int dest_row = idx % m;
			int dest_col = idx/m;
			int src_row = dest_row;
			int src_col = (n - dest_col) - 1;
			d_q_reversed[idx] = d_q[src_col * m + src_row];
		} );
	}

	/**
	 * Truncate Q transpose to top k
	 *
	 * @param Qt
	 * @param QtTrunc
	 * @param context
	 */
	template<typename T>
	void row_reverse_trunc_q(const Matrix<T> &Qt, Matrix<T> &QtTrunc, DeviceContext &context){
		auto m = Qt.rows();
		auto k = QtTrunc.rows();
		auto d_q = Qt.data();
		auto d_q_trunc = QtTrunc.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+QtTrunc.size(), [=]__device__(int idx){
			int dest_row = idx % k;
			int dest_col = idx / k;
			int src_row = (m - dest_row) - 1;
			int src_col = dest_col;
			float q = d_q[src_col * m + src_row];
			d_q_trunc[idx] = q;
		} );
	}

	/**
	 * Calculate the U matrix, which is defined as:
	 * U = A*V/sigma where A is our X Matrix, V is Q, and sigma is 1/w_i
	 *
	 * @param X
	 * @param Q
	 * @param w
	 * @param U
	 * @param context
	 */
	template<typename T>
	void calculate_u(const Matrix<T> &X, const Matrix<T> &Q, const Matrix<T> &w, Matrix<T> &U, DeviceContext &context){
		multiply(X, Q, U, context, false, false, 1.0f); //A*V
		auto d_u = U.data();
		auto d_sigma = w.data();
		auto column_size = U.rows();
		auto counting = thrust::make_counting_iterator <int>(0);
		thrust::for_each(counting, counting+U.size(), [=]__device__(int idx){
			int column = idx/column_size;
			float sigma = d_sigma[column];
			float u = d_u[idx];
			if(sigma != 0.0){
				d_u[idx] = u * 1.0/sigma;
			} else{
				d_u[idx] = 0.0;
			}
		} );

	}

	/**
	 * Obtain SVD attributes, which are as follows:
	 * 1.Singular Values
	 * 2.U matrix
	 * 3.Explained Variance
	 * 4.Explained Variance Ratio
	 */
	template<typename T>
	void get_tsvd_attr(Matrix<T> &X, Matrix<T> &Q, double* _Q, Matrix<T> &w, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param, DeviceContext &context){

		//Obtain Q^T to obtain vector as row major order
		Matrix<T>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context); //Needed for calculate_u()
		Matrix<T>QtTrunc(_param.k, Qt.columns());
		row_reverse_trunc_q(Qt, QtTrunc, context);
		QtTrunc.copy_to_host(_Q); //Send to host

		//Obtain square root of eigenvalues, which are singular values
		T generic_zero = 0.0;
		w.transform([=]__device__(T elem){
			if(elem > generic_zero){
				return std::sqrt(elem);
			}else{
				return generic_zero;
			}
		}
		);

		//Sort from biggest singular value to smallest
		std::vector<double> w_temp(w.size());
		w.copy_to_host(w_temp.data()); //Send to host
		std::reverse(w_temp.begin(), w_temp.end());
		std::copy(w_temp.begin(), w_temp.begin() + _param.k, _w);
		Matrix<T>sigma(_param.k, 1);
		sigma.copy(w_temp.data());

		//Get U matrix
		Matrix<T>U(X.rows(), _param.k);
		Matrix<T>QReversed(Q.rows(), Q.columns());
		col_reverse_q(Q, QReversed, context);
		calculate_u(X, QReversed, sigma, U, context);
		U.copy_to_host(_U); //Send to host

		//Explained Variance
		Matrix<T>UmultSigma(U.rows(), U.columns());
		//U * Sigma
		multiply_diag(U, sigma, UmultSigma, context, false);

		Matrix<T>UOnesSigma(UmultSigma.rows(), 1);
		UOnesSigma.fill(1.0f);
		Matrix<T>USigmaVar(_param.k, 1);
		Matrix<T>USigmaColMean(_param.k, 1);
		multiply(UmultSigma, UOnesSigma, USigmaColMean, context, true, false, 1.0f);
		float m_usigma = UmultSigma.rows();
		multiply(USigmaColMean, 1/m_usigma, context);
		calc_var_numerator(UmultSigma, USigmaColMean, USigmaVar, context);
		multiply(USigmaVar, 1/m_usigma, context);
		USigmaVar.copy_to_host(_explained_variance);

		//Explained Variance Ratio
		//Set aside matrix of 1's for getting sum of columnar variances
		Matrix<T>XmultOnes(X.rows(), 1);
		XmultOnes.fill(1.0f);
		Matrix<T>XVar(X.columns(), 1);
		Matrix<T>XColMean(X.columns(), 1);
		multiply(X, XmultOnes, XColMean, context, true, false, 1.0f);
		float m = X.rows();
		multiply(XColMean, 1/m, context);
		calc_var_numerator(X, XColMean, XVar, context);
		multiply(XVar, 1/m, context);

		Matrix<T>XVarSum(1,1);
		multiply(XVar, XmultOnes, XVarSum, context, true, false, 1.0f);
		Matrix<T>ExplainedVarRatio(_param.k, 1);
		divide(USigmaVar, XVarSum, ExplainedVarRatio, context);
		ExplainedVarRatio.copy_to_host(_explained_variance_ratio);
	}


	void outer_product(Matrix<float>& A, float eigen_value, const Matrix<float>& eigen_vector, const Matrix<float>& eigen_vector_transpose, DeviceContext& context)
	{
		safe_cublas(cublasSger(context.cublas_handle, A.rows(), A.columns(), &eigen_value, eigen_vector.data(), 1, eigen_vector_transpose.data(), 1, A.data(), A.rows()));
	}

	void outer_product(Matrix<double>& A, double eigen_value, const Matrix<double>& eigen_vector, const Matrix<double>& eigen_vector_transpose, DeviceContext& context)
	{
		safe_cublas(cublasDger(context.cublas_handle, A.rows(), A.columns(), &eigen_value, eigen_vector.data(), 1, eigen_vector_transpose.data(), 1, A.data(), A.rows()));
	}

	/**
	 * Conduct truncated svd using cusolverDnSsyevd
	 *
	 * @param X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	template<typename T>
	void cusolver_tsvd(Matrix<T> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param){
		//Allocate matrix for X^TX
		Matrix<T>XtX(_param.X_n, _param.X_n);

		//Create context
		DeviceContext context;

		//Multiply X and Xt and output result to XtX
		multiply(X, X, XtX, context, true, false, 1.0f);

		//Set up Q (V^T) and w (singular value) matrices (w is a matrix of size Q.rows() by 1; really just a vector
		Matrix<T>Q(XtX.rows(), XtX.columns()); // n X n -> V^T
		Matrix<T>w(Q.rows(), 1);
		calculate_eigen_pairs_exact(XtX, Q, w, context);

		//Get tsvd attributes
		get_tsvd_attr(X, Q, _Q, w, _w, _U, _explained_variance, _explained_variance_ratio, _param, context);
	}

	/**
	 * Conduct truncated svd using the power method
	 *
	 * @param X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	template<typename T>
	void power_tsvd(Matrix<T> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param){
		//Allocate matrix for X^TX
		Matrix<T>M(_param.X_n, _param.X_n);

		//Allocate matrix to be used in outer product calculation
		Matrix<T>A(M.rows(), M.columns());

		//Create context
		DeviceContext context;

		//Multiply X and Xt and output result to XtX
		multiply(X, X, M, context, true, false, 1.0f);

		//Set up Q (V^T) and w (singular value) matrices (w is a matrix of size Q.rows() by 1; really just a vector
		Matrix<T>Q(M.rows(), _param.k);
		Matrix<T>w(_param.k, 1);
		std::vector<T> w_temp(w.size());

		/* Power Method for finding all eigenvectors/eigen values
		 * Logic:
		 *
		 * Since the matrix is symmetric, there exists an orthogonal basis of eigenvectors. Once you have found an eigenvector, extend it to an orthogonal basis,
		 * rewrite the matrix in terms of this basis and restrict to the orthogonal space of the known eigenvector.
		 * This is comprised in the method of deflation:
		 *     You first find the first eigenvector v1 (with a maximal λ1), by iterating xn+1=Axn/|Axn|with a "random" initial x0. Once you have found a good approximation
		 *     for v1, you consider B=A−λ1|v1|2 * v1vT (this simple step replaces the "rewrite in terms of this basis" above). This is a matrix that behaves like A for anything
		 *     orthogonal to v1 and zeroes out v1. Use the power method on B again, which will reveal v2, an eigenvector of a largest eigenvalue of B.
		 *     Then switch to C=B−λ2|v2|2 * v2vT so on.
		 */
		Matrix<T>b_k(_param.X_n, 1);
		Matrix<T>b_k1(_param.X_n, 1);

		for(int i = 0; i < _param.k; i ++){
			//Set aside vector of randoms (n x 1)
			b_k.random(_param.random_state + i);
			T previous_eigenvalue_estimate = FLT_MAX;
			T eigen_value_estimate = FLT_MAX;
			for(int iter=0; iter<_param.n_iter;iter++){
				//fprintf(stderr,"k=%d/%d iter=%d/%d\n",i,_param.k,iter,_param.n_iter); fflush(stderr);
				multiply(M, b_k, b_k1, context);
				dot_product(b_k1, b_k, eigen_value_estimate, context);
				if(std::abs(eigen_value_estimate - previous_eigenvalue_estimate) <= (_param.tol * std::abs(previous_eigenvalue_estimate))) {
					break;
				}
				normalize_vector_cublas(b_k1, context);
				b_k.copy(b_k1);
				previous_eigenvalue_estimate = eigen_value_estimate;
			}
			//Obtain eigen value
			w_temp[i] = eigen_value_estimate;

			//Put eigen vector into (starting at last column of Q)
			thrust::copy(b_k.dptr(), b_k.dptr()+b_k.size(), Q.dptr()+Q.rows()*(Q.columns()-i-1));

			//Get rid of eigen effect from original matrix (deflation)
			//multiply(A, 0.0, context);
			A.zero();
			outer_product(A, eigen_value_estimate, b_k, b_k, context);
			subtract(M, A, M, context);
		}
		//Fill in w from vector w_temp
		std::reverse(w_temp.begin(), w_temp.end());
		w.copy(w_temp.data());

		//Get tsvd attributes
		get_tsvd_attr(X, Q, _Q, w, _w, _U, _explained_variance, _explained_variance_ratio, _param, context);

	}
}

template void tsvd::cusolver_tsvd<double>(Matrix<double> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::power_tsvd<double>(Matrix<double> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::cusolver_tsvd<float>(Matrix<float> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::power_tsvd<float>(Matrix<float> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);


