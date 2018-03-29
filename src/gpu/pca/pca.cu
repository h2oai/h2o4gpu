#include "pca.h"
#include "../tsvd/utils.cuh"
#include "../data/matrix.cuh"
#include "../device/device_context.cuh"
#include "../tsvd/tsvd.h"

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
void pca(const double* _X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, double* _mean, params _param) {
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

        tsvd::params svd_param = {_param.X_n, _param.X_m, _param.k, _param.algorithm, _param.verbose, _param.gpu_id};

        tsvd::truncated_svd_matrix(XCentered, _Q, _w, _U, _explained_variance, _explained_variance_ratio, svd_param);

        if(_param.whiten) {
            // TODO whiten
        }

	} catch (const std::exception &e) {
	    std::cerr << "pca error: " << e.what() << "\n";
	} catch (std::string e) {
	    std::cerr << "pca error: " << e << "\n";
	} catch (...) {
		std::cerr << "pca error\n";
	}
}

}
