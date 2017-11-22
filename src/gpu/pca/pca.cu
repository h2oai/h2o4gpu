#include "../data/matrix.cuh"
#include "../device/device_context.cuh"
#include "pca.h"
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
void pca(const double* _X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param) {
	try {

		//Take in X matrix and allocate for X^TX
		tsvd::Matrix<float>X(_param.X_m, _param.X_n);
		X.copy(_X);

		tsvd::Matrix<float>XtX(_param.X_n, _param.X_n);

		//create context
		tsvd::DeviceContext context;

        tsvd::normalize_columns(X, context);

        tsvd::params svd_param = {_param.X_n, _param.X_m, _param.k};

        tsvd::truncated_svd_matrix(X, _Q, _w, _U, _explained_variance, _explained_variance_ratio, svd_param);

        if(_param.whiten) {
            // TODO whiten
        }

	} catch (const std::exception &e) {
	    std::cerr << "tsvd error: " << e.what() << "\n";
	} catch (std::string e) {
	    std::cerr << "tsvd error: " << e << "\n";
	} catch (...) {
		std::cerr << "tsvd error\n";
	}
}

}
