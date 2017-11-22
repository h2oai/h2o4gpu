#pragma once
#include "../tsvd/utils.cuh"
#include "../device/device_context.cuh"
#include "cusolverDn.h"
#include <../../../cub/cub/cub.cuh>

namespace tsvd
{

	typedef float  tsvd_float;

	/**
	 * \class	Matrix
	 *
	 * \brief	Matrix type. Stores data internally in column major fortsvd.
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
		 * \brief	Constructor. Initialize tsvdrix with m rows and n columns in device memory.
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
		 * \brief	Constructor. Initialise tsvdrix by copying existing tsvdrix.
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
		 * \brief	Get thrust device pointer to tsvdrix data. Useful for invoking thrust functions.
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
		 * \brief	Get const thrust device pointer to tsvdrix data. Useful for invoking thrust functions.
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
		 * \return	Number of tsvdrix rows.
		 */

		size_t rows() const
		{
			return _m;
		}

		/**
		 * \fn	size_t columns() const
		 *
		 * \return	Number of tsvdrix columns.
		 */

		size_t columns() const
		{
			return _n;
		}

		/**
		 * \fn	size_t size() const
		 *
		 * \return Number of tsvdrix elements (m*n).
		 */

		size_t size() const
		{
			return _n * _m;
		}

		/**
		 * \fn	void zero()
		 *
		 * \brief	Zeroes tsvdrix elements.
		 *
		 */

		void zero()
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, 0);
		}

		/**
		 * \fn	void fill(T val)
		 *
		 * \brief	Fills tsvdrix with given value.
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
		 * \brief	Fills tsvdrix elements with uniformly distributed numbers between 0-1.0
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
				                  thrust::uniform_real_distribution<tsvd_float> uniDist;
				                  randEng.discard(idx);
				                  return uniDist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void random_normal(int random_seed = 0)
		 *
		 * \brief	Fill tsvdrix with normally distributed random numbers between zero and one.
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
				                  thrust::normal_distribution<tsvd_float> dist;
				                  randEng.discard(idx);
				                  return dist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void copy(const T*hptr)
		 *
		 * \brief	Copies from host pointer to tsvdrix. Assumes host pointer contains array of same size as tsvdrix.
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
			tsvd_check(M.rows() == this->rows()&&M.columns() == this->columns(), "Cannot copy tsvdrix. Dimensions are different.");
			thrust::copy(M.dptr(), M.dptr() + M.size(), this->dptr());
		}


		void print() const
		{
			thrust::host_vector<T> h_tsvd(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data + _n * _m));
			for (auto i = 0; i < _m; i++)
			{
				for (auto j = 0; j < _n; j++)
				{
					printf("%1.2f ", h_tsvd[j * _m + i]);
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

	void multiply_diag(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context, bool left_diag);

	/**
	 * \fn	void multiply(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false, tsvd_float alpha=1.0f);
	 *
	 * \brief	Matrix multiplication. ABa = C. A or B may be transposed. a is a scalar.
	 *
	 * \param 		  	A		   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B		   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	C		   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	   	The context.
	 * \param 		  	transpose_a	(Optional) True to transpose a.
	 * \param 		  	transpose_b	(Optional) True to transpose b.
	 * \param 		  	alpha	   	(Optional) The alpha.
	 */

	void multiply(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false, tsvd_float alpha = 1.0f);

	/**
	 * \fn	void multiply(Matrix<tsvd_float>& A, const tsvd_float a ,DeviceContext& context);
	 *
	 * \brief	Matrix scalar multiplication.
	 *
	 * \param [in,out]	A	   	The Matrix&lt;float&gt; to process.
	 * \param 		  	a	   	The scalar.
	 * \param [in,out]	context	The context.
	 */

	void multiply(Matrix<tsvd_float>& A, const tsvd_float a, DeviceContext& context);

	/**
	 * \fn	void tsvdrix_sub(const Matrix<tsvd_float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context)
	 *
	 * \brief	Matrix subtraction. A - B = C.
	 *
	 */

	void subtract(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context);

	/**
	 * \fn	void add(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context);
	 *
	 * \brief	Matrix addition. A + B = C	
	 *
	 * \param 		  	A	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	C	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void add(const Matrix<tsvd_float>& A, const Matrix<tsvd_float>& B, Matrix<tsvd_float>& C, DeviceContext& context);
	/**
	 * \fn	void transpose(const Matrix<tsvd_float >&A, Matrix<tsvd_float >&B, DeviceContext& context)
	 *
	 * \brief	Transposes tsvdrix A into tsvdrix B.
	 *
	 * \param 		  	A	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	B	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void transpose(const Matrix<tsvd_float>& A, Matrix<tsvd_float>& B, DeviceContext& context);

	/**
	 * \fn	void linear_solve(const Matrix<tsvd_float>& A, Matrix<tsvd_float>& X, const Matrix<tsvd_float>& B, DeviceContext& context)
	 *
	 * \brief	Solve linear system AX=B to find B.
	 *
	 * \param 		  	A	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	X	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void linear_solve(const Matrix<tsvd_float>& A, Matrix<tsvd_float>& X, const Matrix<tsvd_float>& B, DeviceContext& context);

	/**
	 * \fn	void pseudoinverse(const Matrix<tsvd_float>& A, Matrix<tsvd_float>& pinvA, DeviceContext& context)
	 *
	 * \brief	Calculate Moore-Penrose seudoinverse using the singular value decomposition method.
	 *
	 * \param 		  	A	   	Input tsvdrix.
	 * \param [in,out]	pinvA  	The pseudoinverse out.
	 * \param [in,out]	context	Device context.
	 */

	void pseudoinverse(const Matrix<tsvd_float>& A, Matrix<tsvd_float>& pinvA, DeviceContext& context);

	/**
	 * \fn	void normalize_columns(Matrix<tsvd_float>& M, Matrix<tsvd_float>& M_temp, Matrix<tsvd_float>& column_length, Matrix<tsvd_float>& ones, DeviceContext& context);
	 *
	 * \brief	Normalize tsvdrix columns.
	 *
	 * \param [in,out]	M			 	The Matrix&lt;tsvd_float&gt; to process.
	 * \param [in,out]	M_temp		 	Temporary storage tsvdrix of size >= M.
	 * \param [in,out]	column_length	Temporary storage tsvdrix with one element per column.
	 * \param [in,out]	ones		 	Matrix of ones of length M.columns().
	 * \param [in,out]	context		 	The context.
	 */

	void normalize_columns(Matrix<tsvd_float>& M, Matrix<tsvd_float>& M_temp, Matrix<tsvd_float>& column_length, const Matrix<tsvd_float>& ones, DeviceContext& context);

	void normalize_columns(Matrix<tsvd_float>& M, DeviceContext& context);

	/**
	 * \fn	void residual(const Matrix<tsvd_float >&X, const Matrix<tsvd_float >&D, const Matrix<tsvd_float >&S, Matrix<tsvd_float >&R, DeviceContext & context);
	 *
	 * \brief	Calculate residual R = X - DS
	 *
	 */

	void residual(const Matrix<tsvd_float>& X, const Matrix<tsvd_float>& D, const Matrix<tsvd_float>& S, Matrix<tsvd_float>& R, DeviceContext& context);

	void calculate_eigen_pairs_exact(const Matrix<tsvd_float>& X, Matrix<tsvd_float>& Q, Matrix<tsvd_float>& w, DeviceContext& context);


}
