/*************************************************************************
 *
 * Copyright (c) 2018, H2O.ai, Inc. All rights reserved.
 *
 ************************************************************************/

#pragma once

#ifdef WIN32
#define matrixfactorization_export __declspec(dllexport)
#else
#define matrixfactorization_export
#endif

namespace sparse {
	struct TestTrainDataHeader;
}

namespace matrixfactorization {

template<typename Type=float, bool Multithreaded=true>
void process_data(const char* test_data_file,
		const char* train_data_file,
		const char* output_folder,
		sparse::TestTrainDataHeader& parameters);
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
		int THETA_BATCH=3,
		int n_iter=10,
		int verbose=300,
		int gpu_id=0);

} // namespace matrixfactorization
