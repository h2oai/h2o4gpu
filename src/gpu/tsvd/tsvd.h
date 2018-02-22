#pragma once

#ifdef WIN32
#define tsvd_export __declspec(dllexport)
#else
#define tsvd_export
#endif

#include "../data/matrix.cuh"

namespace tsvd


{

	extern "C"
	{
		struct params
		{
			int X_n;
			int X_m;
			int k;
			const char* algorithm;
			int n_iter;
			int random_state;
			float tol;
			int verbose;
			int gpu_id;
		};

		/**
		 *
		 * \param 		  	_X
		 * \param [in,out]	_Q
		 * \param [in,out]	_w
		 * \param [in,out]  _U
		 * \param [out] 	_explained_variance
		 * \param[out]		_explained_variance_ratio
		 * \param 		  	_param
		 */

		tsvd_export void truncated_svd(const double * _X, double * _Q, double * _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);

	}

	template<typename T>
	void cusolver_tsvd(Matrix<T> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
	template<typename T>
	void power_tsvd(Matrix<T> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
	void outer_product(Matrix<float>& A, float eigen_value, const Matrix<float>& eigen_vector, const Matrix<float>& eigen_vector_transpose, DeviceContext& context);
	void outer_product(Matrix<double>& A, float eigen_value, const Matrix<double>& eigen_vector, const Matrix<double>& eigen_vector_transpose, DeviceContext& context);

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
	template<typename T>
	void truncated_svd(const double* _X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param)
	{
		safe_cuda(cudaSetDevice(_param.gpu_id));
		Matrix<T>X(_param.X_m, _param.X_n);
		X.copy(_X);
		truncated_svd_matrix(X, _Q, _w, _U, _explained_variance, _explained_variance_ratio, _param);
	}

	template<typename T>
	void truncated_svd_matrix(Matrix<T> &X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param)
	{
		std::string algorithm(_param.algorithm);
		try
		{
			if(algorithm == "cusolver"){
				if(_param.verbose==1){
				 fprintf(stderr,"Algorithm is cusolver with k = %d\n",_param.k); fflush(stderr);
				}
				tsvd::cusolver_tsvd(X, _Q, _w, _U, _explained_variance, _explained_variance_ratio, _param);
			}
			else {
				if(_param.verbose==1){
				 fprintf(stderr,"Algorithm is power with k = %d and number of iterations = %d\n",_param.k,_param.n_iter); fflush(stderr);
				}
				tsvd::power_tsvd(X, _Q, _w, _U, _explained_variance, _explained_variance_ratio, _param);
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

	tsvd_export void truncated_svd_matrix(Matrix<float> &X, double * _Q, double * _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
	tsvd_export void truncated_svd_matrix(Matrix<double> &X, double * _Q, double * _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);

}
