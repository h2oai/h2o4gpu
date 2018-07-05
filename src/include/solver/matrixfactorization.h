/*
 * matrixfactorization.h
 *
 *  Created on: Jul 4, 2018
 *      Author: monika
 */

#ifndef SRC_INCLUDE_SOLVER_MATRIXFACTORIZATION_H_
#define SRC_INCLUDE_SOLVER_MATRIXFACTORIZATION_H_

#ifdef WIN32
#define matrixfactorization_export __declspec(dllexport)
#else
#define matrixfactorization_export
#endif

namespace matrixfactorization {

/**
 * @param [in]		train_data_file
 * @param [in] 		test_data_file
 * @param [in] 		output_folder
 * @param [in,out]	thetaTHost
 * @param [in,out]	XTHost
 * @param [in]		F
 * @param [in]		lambda
 * @param [in]		X_BATCH
 * @param [in]		THETA_BATCH
 */

matrixfactorization_export void matrixfactorization_float(
		const char* test_data_file,
		const char* train_data_file,
		const char* output_folder,
		float *thetaTHost,
		float* XTHost,
		int F=100,
		float lambda=0.048,
		int X_BATCH=1,
		int THETA_BATCH=3);

} // namespace matrixfactorization



#endif /* SRC_INCLUDE_SOLVER_MATRIXFACTORIZATION_H_ */
