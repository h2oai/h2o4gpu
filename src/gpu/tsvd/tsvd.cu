#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "../device/device_context.cuh"
#include "tsvd.h"
#include <ctime>
#include <thrust/iterator/counting_iterator.h>
#include<algorithm>
#include <thrust/sequence.h>

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
void divide(const Matrix<float> &XVar, const Matrix<float> &XVarSum, Matrix<float> &ExplainedVarRatio, DeviceContext &context){
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
void square_val(const Matrix<float> &UmultSigma, Matrix<float> &UmultSigmaSquare, DeviceContext &context){
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
void calc_var(const Matrix<float>UmultSigma, int k, Matrix<float> &UmultSigmaVar, DeviceContext &context){
	//Set aside matrix of 1's for getting columnar sums(t(UmultSima) * UmultOnes)
	Matrix<float>UmultOnes(UmultSigma.rows(), 1);
	UmultOnes.fill(1.0f);

	//Allocate matrices for variance calculation
	Matrix<float>UmultSigmaSquare(UmultSigma.rows(), UmultSigma.columns());
	Matrix<float>UmultSigmaSum(k, 1);
	Matrix<float>UmultSigmaSumSquare(k, 1);
	Matrix<float>UmultSigmaSumOfSquare(k, 1);
	Matrix<float>UmultSigmaVarNum(k, 1);

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
void calc_var_numerator(const Matrix<float> &X, const Matrix<float> &UColMean, Matrix<float> &UVar, DeviceContext &context){
	auto m = X.rows();
	variance_iterator<float> variance(X.data(), UColMean.data(), m);
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
void col_reverse_q(const Matrix<float> &Q, Matrix<float> &QReversed, DeviceContext &context){
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
void row_reverse_trunc_q(const Matrix<float> &Qt, Matrix<float> &QtTrunc, DeviceContext &context){
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
void calculate_u(const Matrix<float> &X, const Matrix<float> &Q, const Matrix<float> &w, Matrix<float> &U, DeviceContext &context){
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
 * Conduct truncated SVD on a matrix
 *
 * @param _X
 * @param _Q
 * @param _w
 * @param _U
 * @param _explained_variance
 * @param _explained_variance_ratio
 * @param _param
 */
void truncated_svd(const double* _X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param)
{
    Matrix<float>X(_param.X_m, _param.X_n);
	X.copy(_X);
    truncated_svd(_X, _Q, _w, _U, _explained_variance, _explained_variance_ratio, _param);
}

void truncated_svd_matrix(Matrix<float> X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param)
{
	try
	{
		//Allocate matrix for X^TX
		Matrix<float>XtX(_param.X_n, _param.X_n);

		//create context
		DeviceContext context;

		//Multiply X and Xt and output result to XtX
		multiply(X, X, XtX, context, true, false, 1.0f);

		//Set up Q (V^T) and w (singular value) matrices (w is a matrix of size Q.rows() by 1; really just a vector
		Matrix<float>Q(XtX.rows(), XtX.columns()); // n X n -> V^T
		Matrix<float>w(Q.rows(), 1);
		calculate_eigen_pairs_exact(XtX, Q, w, context);

		//Obtain Q^T to obtain vector as row major order
		Matrix<float>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context); //Needed for calculate_u()
		Matrix<float>QtTrunc(_param.k, Qt.columns());
		row_reverse_trunc_q(Qt, QtTrunc, context);
		QtTrunc.copy_to_host(_Q); //Send to host

		//Obtain square root of eigenvalues, which are singular values
		w.transform([=]__device__(float elem){
			if(elem > 0.0){
				return std::sqrt(elem);
			}else{
				return 0.0f;
			}
		}
		);

		//Sort from biggest singular value to smallest
		std::vector<double> w_temp(w.size());
		w.copy_to_host(w_temp.data()); //Send to host
		std::reverse(w_temp.begin(), w_temp.end());
		std::copy(w_temp.begin(), w_temp.begin() + _param.k, _w);
		Matrix<float>sigma(_param.k, 1);
		sigma.copy(w_temp.data());

		//Get U matrix
		Matrix<float>U(X.rows(), _param.k);
		Matrix<float>QReversed(Q.rows(), Q.columns());
		col_reverse_q(Q, QReversed, context);
		calculate_u(X, QReversed, sigma, U, context);
		U.copy_to_host(_U); //Send to host

		//Explained Variance
		Matrix<float>UmultSigma(U.rows(), U.columns());
		//U * Sigma
		multiply_diag(U, sigma, UmultSigma, context, false);

		Matrix<float>UOnesSigma(UmultSigma.rows(), 1);
		UOnesSigma.fill(1.0f);
		Matrix<float>USigmaVar(_param.k, 1);
		Matrix<float>USigmaColMean(_param.k, 1);
		multiply(UmultSigma, UOnesSigma, USigmaColMean, context, true, false, 1.0f);
		float m_usigma = UmultSigma.rows();
		multiply(USigmaColMean, 1/m_usigma, context);
		calc_var_numerator(UmultSigma, USigmaColMean, USigmaVar, context);
		multiply(USigmaVar, 1/m_usigma, context);
		USigmaVar.copy_to_host(_explained_variance);

		//Explained Variance Ratio
		//Set aside matrix of 1's for getting sum of columnar variances
		Matrix<float>XmultOnes(X.rows(), 1);
		XmultOnes.fill(1.0f);
		Matrix<float>XVar(X.columns(), 1);
		Matrix<float>XColMean(X.columns(), 1);
		multiply(X, XmultOnes, XColMean, context, true, false, 1.0f);
		float m = X.rows();
		multiply(XColMean, 1/m, context);
		calc_var_numerator(X, XColMean, XVar, context);
		multiply(XVar, 1/m, context);

		Matrix<float>XVarSum(1,1);
		multiply(XVar, XmultOnes, XVarSum, context, true, false, 1.0f);
		Matrix<float>ExplainedVarRatio(_param.k, 1);
		divide(USigmaVar, XVarSum, ExplainedVarRatio, context);
		ExplainedVarRatio.copy_to_host(_explained_variance_ratio);

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
