#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "../device/device_context.cuh"
#include "../../include/solver/tsvd.h"
#include <ctime>
#include <thrust/iterator/counting_iterator.h>
#include<algorithm>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include "../data/matrix.cuh"

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
	 * Utility to calculate variance for each column of a matrix
	 *
	 * @param X
	 * @param UColMean
	 * @param UVar
	 * @param context
	 */
	template<typename T>
	void calc_var_numerator(Matrix<T> &X, const Matrix<T> &UColMean, Matrix<T> &UVar, DeviceContext &context){
		auto counting = thrust::make_counting_iterator(0);
		auto d_X = X.data();
		auto d_Mean = UColMean.data();
		auto rows = X.rows();
		thrust::for_each(counting, counting+X.size(),  [=] __device__ (size_t idx)
		{
			auto column = idx/rows;
			d_X[idx] = std::pow(d_X[idx] - d_Mean[column],2);
		});

		Matrix<T>Ones(X.rows(), 1);
		Ones.fill(1.0f);
		multiply(X, Ones, UVar, context, true, false, 1.0f);

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
	 * Transform matrix into absolute values
	 *
	 * @param UmultSigma
	 * @param UmultSigmaSquare
	 * @param context
	 */
	template<typename T>
	void get_abs(const Matrix<T> &U, Matrix<T> &U_abs, DeviceContext &context){
		thrust::transform(U.dptr(), U.dptr() + U.size(), U_abs.dptr(), [=]__device__(T val){
            return abs(val);
        });
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
	template<typename T, typename S>
	void get_tsvd_attr(Matrix<T> &X, Matrix<T> &Q, S _Q, Matrix<T> &w, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param, DeviceContext &context){

		//Obtain Q^T to obtain vector as row major order
		Matrix<T>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context); //Needed for calculate_u()
		Matrix<T>QtTrunc(_param.k, Qt.columns());
		row_reverse_trunc_q(Qt, QtTrunc, context);

		if (_param.whiten) {
			auto d_q = QtTrunc.data();
			auto x_sqrt_row = std::sqrt(X.rows());
			auto d_sigma = w.data();
			auto column_size = QtTrunc.rows();
			auto counting = thrust::make_counting_iterator <int>(0);
			thrust::for_each(counting, counting+QtTrunc.size(), [=]__device__(int idx){
				int column = idx/column_size;
				T sigma = d_sigma[column];
				T q = d_q[idx];
				d_q[idx] = (q * x_sqrt_row)/std::sqrt(sigma);
			} );
		}

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

		/**
		 * SVD sign correction (same as svd_flip() in sklearn) -> https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py#L499
		   Sign correction to ensure deterministic output from SVD.
    	   Adjusts the columns of u and the rows of v such that the loadings in the
    	   columns in u that are largest in absolute value are always positive.
		 */
		std::vector<int> result_array(U.columns());
		max_index_per_column(U, result_array, context);
		Matrix<T>Signs(1, _param.k);
		thrust::device_vector<int> d_results = result_array;
		auto d_U = U.data();
		auto d_Signs = Signs.data();
		auto counting = thrust::make_counting_iterator <int>(0);
		auto ptr = d_results.data();
		thrust::for_each(counting, counting+Signs.size(), [=]__device__(int idx){
			int u_idx = ptr[idx];
			T val = 1.0;
			if (d_U[u_idx] < 0.0) {
				val = -1.0;
			} else if (d_U[u_idx] == 0.0) {
				val = 0.0;
			} else {
				val = 1.0;
			}
			d_Signs[idx] = val;
		} );
		multiply_diag(U, Signs, U, context, false);
		multiply_diag(QtTrunc, Signs, QtTrunc, context, true);
		U.copy_to_host(_U); //Send to host
		QtTrunc.copy_to_host(_Q); //Send to host

		//U * Sigma (_X_transformed)
		Matrix<T>UmultSigma(U.rows(), U.columns());
		multiply_diag(U, sigma, UmultSigma, context, false);
		UmultSigma.copy_to_host(_X_transformed); //Send to host

		//Explained Variance
		Matrix<T>UOnesSigma(UmultSigma.rows(), 1);
		UOnesSigma.fill(1.0f);
		Matrix<T>USigmaVar(_param.k, 1);
		Matrix<T>USigmaColMean(_param.k, 1);
		multiply(UmultSigma, UOnesSigma, USigmaColMean, context, true, false, 1.0f);
		float m_usigma = UmultSigma.rows();
		multiply(USigmaColMean, 1/m_usigma, context);
		calc_var_numerator(UmultSigma, USigmaColMean, USigmaVar, context);
		multiply(USigmaVar, 1/m_usigma, context);
		USigmaVar.copy_to_host(_explained_variance); //Send to host

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
		ExplainedVarRatio.copy_to_host(_explained_variance_ratio); //Send to host
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
	template<typename T, typename S>
	void cusolver_tsvd(Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param){
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
		get_tsvd_attr(X, Q, _Q, w, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param, context);
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
	template<typename T, typename S>
	void power_tsvd(Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param){
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
				//fprintf(stderr,"k=%d/%d iter=%d/%d\n",i,_param.k-1,iter,_param.n_iter); fflush(stderr);
				multiply(M, b_k, b_k1, context);
				dot_product(b_k1, b_k, &eigen_value_estimate, context);
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
		get_tsvd_attr(X, Q, _Q, w, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param, context);

	}

	/**
	 * Conduct truncated SVD on a matrix with float type
	 *
	 * @param _X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	void truncated_svd_float(const float *_X, float *_Q, float *_w, float *_U, float* _X_transformed, float *_explained_variance, float *_explained_variance_ratio, params _param)
	{
		safe_cuda(cudaSetDevice(_param.gpu_id));
		Matrix<float>X(_param.X_m, _param.X_n);
		X.copy(_X);
		truncated_svd_matrix(X, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param);
	}

	/**
	 * Conduct truncated SVD on a matrix with double type
	 *
	 * @param _X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	void truncated_svd_double(const double *_X, double *_Q, double *_w, double *_U, double* _X_transformed, double *_explained_variance, double *_explained_variance_ratio, params _param)
	{
		safe_cuda(cudaSetDevice(_param.gpu_id));
		Matrix<double>X(_param.X_m, _param.X_n);
		X.copy(_X);
		truncated_svd_matrix(X, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param);
	}

	template<typename T, typename S>
	tsvd_export void truncated_svd_matrix(Matrix<T> &X, S _Q, S _w, S _U, S _X_transformed, S _explained_variance, S _explained_variance_ratio, params _param)
	{
		std::string algorithm(_param.algorithm);
		try
		{
			if(algorithm == "cusolver"){
				if(_param.verbose==1){
				 fprintf(stderr,"Algorithm is cusolver with k = %d\n",_param.k); fflush(stderr);
				}
				tsvd::cusolver_tsvd(X, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param);
			}
			else {
				if(_param.verbose==1){
				 fprintf(stderr,"Algorithm is power with k = %d and number of iterations = %d\n",_param.k,_param.n_iter); fflush(stderr);
				}
				tsvd::power_tsvd(X, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, _param);
			}
		}
		catch (const std::exception &e)
		  {
			std::cerr << "tsvd error: " << e.what() << "\n";
		  }
		catch (std::string e)
		  {
			std::cerr << "tsvd error: " << e << "\n";
		  }
		catch (...)
		  {
			std::cerr << "tsvd error\n";
		  }
	}
}

//Impl for floats and doubles
template void tsvd::truncated_svd_matrix<float>(Matrix<float> &X, float* _Q, float* _w, float* _U, float* _X_transformed, float* _explained_variance, float* _explained_variance_ratio, params _param);
template void tsvd::truncated_svd_matrix<double>(Matrix<double> &X, double* _Q, double* _w, double* _U, double* _X_transformed, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::cusolver_tsvd<double>(Matrix<double> &X, double* _Q, double* _w, double* _U, double* _X_transformed, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::power_tsvd<double>(Matrix<double> &X, double* _Q, double* _w, double* _U, double* _X_transformed, double* _explained_variance, double* _explained_variance_ratio, params _param);
template void tsvd::cusolver_tsvd<float>(Matrix<float> &X, float* _Q, float* _w, float* _U, float* _X_transformed, float* _explained_variance, float* _explained_variance_ratio, params _param);
template void tsvd::power_tsvd<float>(Matrix<float> &X, float* _Q, float* _w, float* _U, float* _X_transformed, float* _explained_variance, float* _explained_variance_ratio, params _param);


