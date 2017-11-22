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

	tsvd_export void truncated_svd_matrix(Matrix<float> _X, double * _Q, double * _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
}
