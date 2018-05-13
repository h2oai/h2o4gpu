
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "eigen_sparse_manager.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "als.h"
#include "host_utilities.h"

#define DEVICEID 0
#define ITERS 10

int use_case_files(sparse::TestTrainDataHeader& header) {
	std::string test_data_file("/home/monika/h2o/src/cuda_cu/monika_AMF/data/test_data.dat");
	typedef Eigen::SparseMatrix<int, Eigen::ColMajor> ColMatrix;
	typedef Eigen::SparseMatrix<int, Eigen::RowMajor> RowMatrix;

	ColMatrix colTestMatrix;

	bool result = sparse::loadFileAsSparseMatrix(colTestMatrix, test_data_file);
	if (!result) {
		std::cout << "Sparse Matrix cannot be loaded from file: " << test_data_file << std::endl;
		return -1;
	}

	// create binaries (todo: without the files later)
	// test data
	sparse::serialize_rows_cols_data(colTestMatrix, "/home/monika/h2o/src/cuda_cu/monika_AMF/data/test");
	header.m = (int)colTestMatrix.rows();
	header.n = (int)colTestMatrix.cols();
	header.nnz_test = sparse::getNNZ(colTestMatrix);

	// train data, temporarily used the same test data
	std::string train_data_file("/home/monika/h2o/src/cuda_cu/monika_AMF/data/test_data.dat");
	ColMatrix colTrainMatrix;
	RowMatrix rowTrainMatrix;

	result = sparse::loadFileAsSparseMatrix(colTrainMatrix, train_data_file);
	if (!result) {
		std::cout << "Sparse Matrix cannot be loaded from file: " << train_data_file << std::endl;
		return -1;
	}
	result = sparse::loadFileAsSparseMatrix(rowTrainMatrix, train_data_file);
	if (!result) {
		std::cout << "Sparse Matrix cannot be loaded from file: " << train_data_file << std::endl;
		return -1;
	}

	header.nnz_train = sparse::getNNZ(colTrainMatrix);

	sparse::serialize_rows_cols_data(colTrainMatrix, "/home/monika/h2o/src/cuda_cu/monika_AMF/data/train_csc");
	sparse::serialize_rows_cols_data(rowTrainMatrix, "/home/monika/h2o/src/cuda_cu/monika_AMF/data/train_csr");
	return 0;
}

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

int main(int argc, char **argv)
{
	sparse::TestTrainDataHeader test_train_data_header;
	int result = use_case_files(test_train_data_header);
	if (result) return result;
	// set input parameters
	test_train_data_header.setF(100);
	test_train_data_header.setLambda(0.048);
	test_train_data_header.setXBatch(1);
	test_train_data_header.setThetaBatch(3);

	cudaSetDevice(DEVICEID);
	int* csrRowIndexHostPtr;

	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (test_train_data_header.m + 1) * sizeof(csrRowIndexHostPtr[0])) );
	int* csrColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, test_train_data_header.nnz_train * sizeof(csrColIndexHostPtr[0])) );
	float* csrValHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, test_train_data_header.nnz_train * sizeof(csrValHostPtr[0])) );
	float* cscValHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscValHostPtr, test_train_data_header.nnz_train * sizeof(cscValHostPtr[0])) );
	int* cscRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr, test_train_data_header.nnz_train * sizeof(cscRowIndexHostPtr[0])) );
	int* cscColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr, (test_train_data_header.n+1) * sizeof(cscColIndexHostPtr[0])) );
	int* cooRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cooRowIndexHostPtr, test_train_data_header.nnz_train * sizeof(cooRowIndexHostPtr[0])) );

	//calculate X from thetaT first, need to initialize thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, test_train_data_header.n * test_train_data_header.f * sizeof(thetaTHost[0])) );

	float* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, test_train_data_header.m * test_train_data_header.f * sizeof(XTHost[0])) );

	//initialize thetaT on host
	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < test_train_data_header.n * test_train_data_header.f; k++)
		thetaTHost[k] = 0.2*((float) rand() / (float)RAND_MAX);
	//CG needs to initialize X as well
	for (int k = 0; k < test_train_data_header.m * test_train_data_header.f; k++)
		XTHost[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);;
	printf("*******start loading training and testing sets to host.\n");

	//testing set
	int* cooRowIndexTestHostPtr = (int *) malloc(
			test_train_data_header.nnz_test * sizeof(cooRowIndexTestHostPtr[0]));
	int* cooColIndexTestHostPtr = (int *) malloc(
			test_train_data_header.nnz_test * sizeof(cooColIndexTestHostPtr[0]));
	float* cooValHostTestPtr = (float *) malloc(test_train_data_header.nnz_test * sizeof(cooValHostTestPtr[0]));

	struct timeval tv0;
	gettimeofday(&tv0, NULL);

	// TODO - indptr implementation
	loadCooSparseMatrixBin( (DATA_DIR + "/R_test_coo.data.bin").c_str(), (DATA_DIR + "/R_test_coo.row.bin").c_str(),
							(DATA_DIR + "/R_test_coo.col.bin").c_str(),
			cooValHostTestPtr, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, test_train_data_header.nnz_test);

	loadCSRSparseMatrixBin( (DATA_DIR + "/R_train_csr.data.bin").c_str(), (DATA_DIR + "/R_train_csr.indptr.bin").c_str(),
							(DATA_DIR + "/R_train_csr.indices.bin").c_str(),
			csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr, test_train_data_header.m, test_train_data_header.nnz_train);

	loadCSCSparseMatrixBin( (DATA_DIR + "/R_train_csc.data.bin").c_str(), (DATA_DIR + "/R_train_csc.indices.bin").c_str(),
							(DATA_DIR +"/R_train_csc.indptr.bin").c_str(),
		cscValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, test_train_data_header.n, test_train_data_header.nnz_train);

	loadCooSparseMatrixRowPtrBin( (DATA_DIR + "/R_train_coo.row.bin").c_str(), cooRowIndexHostPtr, test_train_data_header.nnz_train);

	double t0 = seconds();

	doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
			cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
			cooRowIndexHostPtr, thetaTHost, XTHost,
			cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
			test_train_data_header.m, test_train_data_header.n, test_train_data_header.f, test_train_data_header.nnz_train,
			test_train_data_header.nnz_test, test_train_data_header.lambda,
			ITERS, test_train_data_header.X_BATCH, test_train_data_header.THETA_BATCH, DEVICEID);
	printf("\ndoALS takes seconds: %.3f for F = %d\n", seconds() - t0, test_train_data_header.f);


	return 0;
}
