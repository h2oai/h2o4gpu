#include "../../include/solver/pca.h"
#include "../tsvd/utils.cuh"
#include "../data/matrix.cuh"
#include "../device/device_context.cuh"
#include "../../include/solver/tsvd.h"

namespace pca
{

	/**
	* Conduct PCA on a matrix

	 *
	 * @param _X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	void pca_float(const float *_X, float *_Q, float *_w, float *_U, float* _X_transformed, float *_explained_variance, float *_explained_variance_ratio, float *_mean, params _param) {
		try {

			tsvd::safe_cuda(cudaSetDevice(_param.gpu_id));

			//Take in X matrix and allocate for X^TX
			tsvd::Matrix<float>X(_param.X_m, _param.X_n);
			X.copy(_X);

			tsvd::Matrix<float>XtX(_param.X_n, _param.X_n);

			//create context
			tsvd::DeviceContext context;

			//Get columnar means
			tsvd::Matrix<float>XOnes(X.rows(), 1);
			XOnes.fill(1.0f);
			tsvd::Matrix<float>XColMean(X.columns(), 1);
			tsvd::multiply(X, XOnes, XColMean, context, true, false, 1.0f);
			float m = X.rows();
			multiply(XColMean, 1/m, context);
			XColMean.copy_to_host(_mean);

			//Center matrix
			tsvd::Matrix<float>OnesXMeanTranspose(X.rows(), X.columns());
			tsvd::multiply(XOnes, XColMean, OnesXMeanTranspose, context, false, true, 1.0f);
			tsvd::Matrix<float>XCentered(X.rows(), X.columns());
			tsvd::subtract(X, OnesXMeanTranspose, XCentered, context);

			tsvd::params svd_param = {_param.X_n, _param.X_m, _param.k, _param.algorithm, _param.n_iter, _param.random_state, _param.tol, _param.verbose, _param.gpu_id, _param.whiten};

			tsvd::truncated_svd_matrix(XCentered, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, svd_param);

		} catch (const std::exception &e) {
			std::cerr << "pca error: " << e.what() << "\n";
		} catch (std::string e) {
			std::cerr << "pca error: " << e << "\n";
		} catch (...) {
			std::cerr << "pca error\n";
		}
	}

	/**
	 * Conduct PCA on a matrix

	 *
	 * @param _X
	 * @param _Q
	 * @param _w
	 * @param _U
	 * @param _explained_variance
	 * @param _explained_variance_ratio
	 * @param _param
	 */
	void pca_double(const double *_X, double *_Q, double *_w, double *_U, double* _X_transformed, double *_explained_variance, double *_explained_variance_ratio, double *_mean, params _param) {
		try {

			tsvd::safe_cuda(cudaSetDevice(_param.gpu_id));

			//Take in X matrix and allocate for X^TX
			tsvd::Matrix<double>X(_param.X_m, _param.X_n);
			X.copy(_X);

			tsvd::Matrix<double>XtX(_param.X_n, _param.X_n);

			//create context
			tsvd::DeviceContext context;

			//Get columnar means
			tsvd::Matrix<double>XOnes(X.rows(), 1);
			XOnes.fill(1.0f);
			tsvd::Matrix<double>XColMean(X.columns(), 1);
			tsvd::multiply(X, XOnes, XColMean, context, true, false, 1.0f);
			float m = X.rows();
			multiply(XColMean, 1/m, context);
			XColMean.copy_to_host(_mean);

			//Center matrix
			tsvd::Matrix<double>OnesXMeanTranspose(X.rows(), X.columns());
			tsvd::multiply(XOnes, XColMean, OnesXMeanTranspose, context, false, true, 1.0f);
			tsvd::Matrix<double>XCentered(X.rows(), X.columns());
			tsvd::subtract(X, OnesXMeanTranspose, XCentered, context);

			tsvd::params svd_param = {_param.X_n, _param.X_m, _param.k, _param.algorithm, _param.n_iter, _param.random_state, _param.tol, _param.verbose, _param.gpu_id, _param.whiten};

			tsvd::truncated_svd_matrix(XCentered, _Q, _w, _U, _X_transformed, _explained_variance, _explained_variance_ratio, svd_param);

		} catch (const std::exception &e) {
			std::cerr << "pca error: " << e.what() << "\n";
		} catch (std::string e) {
			std::cerr << "pca error: " << e << "\n";
		} catch (...) {
			std::cerr << "pca error\n";
		}
	}

}
