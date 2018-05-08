#pragma once
#include "../utils/utils.cuh"
#include "../device/device_context.cuh"
#include "cusolverDn.h"
#include <../../../cub/cub/cub.cuh>

namespace matrix
{
	using namespace h2o4gpu;

	/**
	 * \class	Matrix
	 *
	 * \brief	Matrix type. Stores data internally in column major
	 *
	 */

	template <typename T>
	class Matrix
	{
		size_t _m;
		size_t _n;

		T* _data;

	public:

		/**
		 * \fn	Matrix()
		 *
		 * \brief	Default constructor.
		 *
		 */

		Matrix() : _m(0), _n(0), _data(nullptr)
		{
		}

		/**
		 * \fn	Matrix(size_t m, size_t n)
		 *
		 * \brief	Constructor. Initialize matrix with m rows and n columns in device memory.
		 *
		 * \param	m	Matrix rows.
		 * \param	n	Matrix columns.
		 */

		Matrix(size_t m, size_t n) : _m(m), _n(n)
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
		}

		/**
		 * \fn	Matrix(const Matrix<T>& M)
		 *
		 * \brief	Constructor. Initialise matrix by copying existing matrix.
		 *
		 * \param	M	The Matrix&lt;T&gt; to copy.
		 */

		Matrix(const Matrix<T>& M) : _n(M.columns()), _m(M.rows())
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
			this->copy(M);
		}

		~Matrix()
		{
			safe_cuda(cudaFree(_data));
		}

		/**
		 * \fn	void resize(size_t m, size_t n)
		 *
		 * \brief	Resizes.
		 *
		 * \param	m	Matrix rows.
		 * \param	n	Matrix columns.
		 */

		void resize(size_t m, size_t n)
		{
			_m = m;
			_n = n;
			if (_data != nullptr)
			{
				safe_cuda(cudaFree(_data));
			}
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
		}

		/**
		 * \fn	T* data()
		 *
		 * \brief Return raw pointer to data. Data is allocated on device.	
		 *
		 * \return	Raw pointer to Matrix data.
		 */

		T* data()
		{
			return _data;
		}

		/**
		 * \fn	T* data()
		 *
		 * \brief Return const raw pointer to data. Data is allocated on device.	
		 *
		 * \return	Raw pointer to Matrix data.
		 */
		const T* data() const
		{
			return _data;
		}

		/**
		 * \fn	thrust::device_ptr<T> dptr()
		 *
		 * \brief	Get thrust device pointer to matrix data. Useful for invoking thrust functions.
		 *
		 * \return	A thrust::device_ptr&lt;T&gt;
		 */

		thrust::device_ptr<T> dptr()
		{
			return thrust::device_pointer_cast(_data);
		}

		/**
		 * \fn	thrust::device_ptr<T> dptr()
		 *
		 * \brief	Get const thrust device pointer to matrix data. Useful for invoking thrust functions.
		 *
		 * \return	A thrust::device_ptr&lt;T&gt;
		 */

		thrust::device_ptr<const T> dptr() const
		{
			return thrust::device_pointer_cast(_data);
		}

		/**
		 * \fn	size_t rows() const
		 *
		 * \return	Number of matrix rows.
		 */

		size_t rows() const
		{
			return _m;
		}

		/**
		 * \fn	size_t columns() const
		 *
		 * \return	Number of matrix columns.
		 */

		size_t columns() const
		{
			return _n;
		}

		/**
		 * \fn	size_t size() const
		 *
		 * \return Number of matrix elements (m*n).
		 */

		size_t size() const
		{
			return _n * _m;
		}

		/**
		 * \fn	void zero()
		 *
		 * \brief	Zeroes matrix elements.
		 *
		 */

		void zero()
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, 0);
		}

		/**
		 * \fn	void fill(T val)
		 *
		 * \brief	Fills matrix with given value.
		 *
		 * \param	val	The value.
		 */

		void fill(T val)
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, val);
		}

		/**
		 * \fn	void random(int random_seed = 0)
		 *
		 * \brief	Fills matrix elements with uniformly distributed numbers between 0-1.0
		 *
		 * \param	random_seed	(Optional) The random seed.
		 */

		void random(int random_seed = 0)
		{
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + _m * _n,
			                  thrust::device_ptr<T>(_data),
			                  [=]__device__(int idx)
			                  {
				                  thrust::default_random_engine randEng(random_seed);
				                  thrust::uniform_real_distribution<T> uniDist;
				                  randEng.discard(idx);
				                  return uniDist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void random_normal(int random_seed = 0)
		 *
		 * \brief	Fill matrix with normally distributed random numbers between zero and one.
		 *
		 * \param	random_seed	(Optional) The random seed.
		 */

		void random_normal(int random_seed = 0)
		{
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + _m * _n,
			                  thrust::device_ptr<T>(_data),
			                  [=]__device__(int idx)
			                  {
				                  thrust::default_random_engine randEng(random_seed);
				                  thrust::normal_distribution<T> dist;
				                  randEng.discard(idx);
				                  return dist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void copy(const T*hptr)
		 *
		 * \brief	Copies from host pointer to matrix. Assumes host pointer contains array of same size as matrix.
		 *
		 * \param	hptr	Host pointer.
		 */

		template <typename HostT>
		void copy(const HostT* hptr)
		{
			if(std::is_same<HostT, T>::value){
				thrust::copy(hptr, hptr + this->size(), this->dptr());
			}else{
				std::vector<T> temp(hptr, hptr + this->size());
				thrust::copy(temp.begin(), temp.end(), this->dptr());
			}
		}

		/**
		 * \fn	void copy(const Matrix<T>& M)
		 *
		 * \brief	Copies the given M.
		 *
		 * \param	M	The Matrix&lt;T&gt; to process.
		 */

		void copy(const Matrix<T>& M)
		{
			h2o4gpu_check(M.rows() == this->rows()&&M.columns() == this->columns(), "Cannot copy matrix. Dimensions are different.");
			thrust::copy(M.dptr(), M.dptr() + M.size(), this->dptr());
		}


		void print() const
		{
			thrust::host_vector<T> h_matrix(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data + _n * _m));
			for (auto i = 0; i < _m; i++)
			{
				for (auto j = 0; j < _n; j++)
				{
					printf("%1.2f ", h_matrix[j * _m + i]);
				}
				printf("\n");
			}
		}

		template<typename function_t>
		void transform(function_t f)
		{
			thrust::transform(this->dptr(), this->dptr() + this->size(), this->dptr(), f);
		}

		//Copy contents of matrix to host pointer
		template<typename host_ptr_t>
		void copy_to_host(host_ptr_t ptr){
			thrust::copy(this->dptr(), this->dptr() + this->size(), ptr);
		}
	};

	void multiply_diag(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context, bool left_diag);
	void multiply_diag(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context, bool left_diag);

	/**
	 * \fn	void multiply(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context, bool transpose_a = false, bool transpose_b = false, float alpha=1.0f);
	 *
	 * \brief	Matrix multiplication. ABa = C. A or B may be transposed. a is a scalar.
	 *
	 * \param 		  	A		   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B		   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	C		   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	   	The context.
	 * \param 		  	transpose_a	(Optional) True to transpose a.
	 * \param 		  	transpose_b	(Optional) True to transpose b.
	 * \param 		  	alpha	   	(Optional) The alpha.
	 */

	void multiply(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context, bool transpose_a = false, bool transpose_b = false, float alpha = 1.0f);

	/**
	 * \fn	void multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context, bool transpose_a = false, bool transpose_b = false, double alpha=1.0f);
	 *
	 * \brief	Matrix multiplication. ABa = C. A or B may be transposed. a is a scalar.
	 *
	 * \param 		  	A		   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B		   	The Matrix&lt;double&gt; to process.
	 * \param [in,out]	C		   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	   	The context.
	 * \param 		  	transpose_a	(Optional) True to transpose a.
	 * \param 		  	transpose_b	(Optional) True to transpose b.
	 * \param 		  	alpha	   	(Optional) The alpha.
	 */

	void multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, device::DeviceContext& context, bool transpose_a = false, bool transpose_b = false, double alpha = 1.0f);

	/**
	 * \fn	void multiply(Matrix<float>& A, const float a ,device::DeviceContext& context);
	 *
	 * \brief	Matrix scalar multiplication.
	 *
	 * \param [in,out]	A	   	The Matrix&lt;float&gt; to process.
	 * \param 		  	a	   	The scalar.
	 * \param [in,out]	context	The context.
	 */

	template<typename T, typename U>
	void multiply(Matrix<T>& A, const U a, device::DeviceContext& context);

	/**
	 * \fn	void matrix_sub(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context)
	 *
	 * \brief	Matrix subtraction. A - B = C.
	 *
	 */

	template<typename T>
	void subtract(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, device::DeviceContext& context);

	/**
	 * \fn	void add(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, device::DeviceContext& context);
	 *
	 * \brief	Matrix addition. A + B = C	
	 *
	 * \param 		  	A	   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	C	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	template<typename T>
	void add(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, device::DeviceContext& context);

	/**
	 * \fn	void transpose(const Matrix<float >&A, Matrix<float >&B, device::DeviceContext& context)
	 *
	 * \brief	Transposes matrix A into matrix B.
	 *
	 * \param 		  	A	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	B	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void transpose(const Matrix<float>& A, Matrix<float>& B, device::DeviceContext& context);

	/**
	 * \fn	void transpose(const Matrix<double >&A, Matrix<double >&B, device::DeviceContext& context)
	 *
	 * \brief	Transposes matrix A into matrix B.
	 *
	 * \param 		  	A	   	The Matrix&lt;double&gt; to process.
	 * \param [in,out]	B	   	The Matrix&lt;double&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void transpose(const Matrix<double>& A, Matrix<double>& B, device::DeviceContext& context);

	/**
	 * \fn	void normalize_columns(Matrix<float>& M, Matrix<float>& M_temp, Matrix<float>& column_length, Matrix<float>& ones, device::DeviceContext& context);
	 *
	 * \brief	Normalize matrix columns.
	 *
	 * \param [in,out]	M			 	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	M_temp		 	Temporary storage matrix of size >= M.
	 * \param [in,out]	column_length	Temporary storage matrix with one element per column.
	 * \param [in,out]	ones		 	Matrix of ones of length M.columns().
	 * \param [in,out]	context		 	The context.
	 */

	void normalize_columns(Matrix<float>& M, Matrix<float>& M_temp, Matrix<float>& column_length, const Matrix<float>& ones, device::DeviceContext& context);
	void normalize_columns(Matrix<double>& M, Matrix<double>& M_temp, Matrix<double>& column_length, const Matrix<double>& ones, device::DeviceContext& context);

	void normalize_columns(Matrix<float>& M, device::DeviceContext& context);
	void normalize_columns(Matrix<double>& M, device::DeviceContext& context);

	/**
	 * \fn	void normalize_vector_cublas(Matrix<float>& M, device::DeviceContext& context)
	 *
	 * \brief	Normalize a vector utilizing cuBLAS
	 *
	 * \param [in,out]	M	    The vector to process
	 * \param [in,out]	context	Device context.
	 */
	void normalize_vector_cublas(Matrix<float>& M, device::DeviceContext& context);

	/**
	 * \fn	void normalize_vector_cublas(Matrix<double>& M, device::DeviceContext& context)
	 *
	 * \brief	Normalize a vector utilizing cuBLAS
	 *
	 * \param [in,out]	M	    The vector to process
	 * \param [in,out]	context	Device context.
	 */
	void normalize_vector_cublas(Matrix<double>& M, device::DeviceContext& context);


	/**
	 * \fn	void normalize_vector_thrust(Matrix<float>& M, device::DeviceContext& context)
	 *
	 * \brief	Normalize a vector utilizng Thrust
	 *
	 * \param [in,out]	M	    The vector to process
	 * \param [in,out]	context	Device context.
	 */

	template<typename T>
	void normalize_vector_thrust(Matrix<T>& M, device::DeviceContext& context);

	/**
	 * \fn	void residual(const Matrix<float >&X, const Matrix<float >&D, const Matrix<float >&S, Matrix<float >&R, device::DeviceContext & context);
	 *
	 * \brief	Calculate residual R = X - DS
	 *
	 */

	void residual(const Matrix<float>& X, const Matrix<float>& D, const Matrix<float>& S, Matrix<float>& R, device::DeviceContext& context);
	void residual(const Matrix<double>& X, const Matrix<double>& D, const Matrix<double>& S, Matrix<double>& R, device::DeviceContext& context);

	void calculate_eigen_pairs_exact(const Matrix<float>& X, Matrix<float>& Q, Matrix<float>& w, device::DeviceContext& context);
	void calculate_eigen_pairs_exact(const Matrix<double>& X, Matrix<double>& Q, Matrix<double>& w, device::DeviceContext& context);

	void dot_product(Matrix<float>& b_k1, Matrix<float>& b_k, float* eigen_value_estimate, device::DeviceContext& context);
	void dot_product(Matrix<double>& b_k1, Matrix<double>& b_k, double* eigen_value_estimate, device::DeviceContext& context);

	void max_index_per_column(Matrix<float>& A, std::vector<int>& result_array, device::DeviceContext& context);
	void max_index_per_column(Matrix<double>& A, std::vector<int>& result_array, device::DeviceContext& context);

	//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//Stricly floating point operations that are not used

	/**
	 * \fn	void linear_solve(const Matrix<float>& A, Matrix<float>& X, const Matrix<float>& B, device::DeviceContext& context)
	 *
	 * \brief	Solve linear system AX=B to find B.
	 *
	 * \param 		  	A	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	X	   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void linear_solve(const Matrix<float>& A, Matrix<float>& X, const Matrix<float>& B, device::DeviceContext& context);

	/**
	 * \fn	void pseudoinverse(const Matrix<float>& A, Matrix<float>& pinvA, device::DeviceContext& context)
	 *
	 * \brief	Calculate Moore-Penrose seudoinverse using the singular value decomposition method.
	 *
	 * \param 		  	A	   	Input matrix.
	 * \param [in,out]	pinvA  	The pseudoinverse out.
	 * \param [in,out]	context	Device context.
	 */

	void pseudoinverse(const Matrix<float>& A, Matrix<float>& pinvA, device::DeviceContext& context);

}
