#pragma once

#ifdef WIN32
#define pca_export __declspec(dllexport)
#else
#define pca_export
#endif

namespace pca
{
	extern "C"
	{

		typedef float  pca_float;

		struct params
		{
			int X_n;
			int X_m;
			int k;
			bool whiten;
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

		pca_export void pca(const double * _X, double * _Q, double * _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param);
	}
}
