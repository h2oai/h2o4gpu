
#include "als.h"
#include "eigen_sparse_manager.h"
#include "host_utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <stdexcept>
#include <thread>

#define DEBUG 1

#define DEVICEID 0
#define ITERS 10

static bool endsWith(const std::string& str, const std::string& suffix)
{
	return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

static bool startsWith(const std::string& str, const std::string& prefix)
{
	return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

template<typename Type=float, bool Multithreaded=true>
void process_test_train_data(const std::string& test_data_file,
		const std::string& train_data_file,
		const std::string& output_folder,
		sparse::TestTrainDataHeader& parameters,
		int F=100,
		float lambda=0.048,
		int X_BATCH=1,
		int THETA_BATCH=3)
{
	parameters.setF(F);
	parameters.setLambda(lambda);
	parameters.setXBatch(X_BATCH);
	parameters.setThetaBatch(THETA_BATCH);

	typedef Eigen::SparseMatrix<Type, Eigen::ColMajor> CMatrix;
	typedef Eigen::SparseMatrix<Type, Eigen::RowMajor> RMatrix;

	CMatrix colMatrixTest, colMatrixTrain; RMatrix rowMatrixTest, rowMatrixTrain;

	// read test and training data file in files/memory
	bool result = sparse::loadFileAsSparseMatrix<decltype(colMatrixTest), sparse::InputFormatFile::column_row_value>(colMatrixTest, test_data_file);
	if (!result) throw std::invalid_argument("Column Sparse Matrix cannot be loaded for test data file!");
	result = sparse::loadFileAsSparseMatrix(rowMatrixTest, test_data_file);
	if (!result) throw std::invalid_argument("Row Sparse Matrix cannot be loaded for test data file!");

	result = sparse::loadFileAsSparseMatrix(colMatrixTrain, train_data_file);
	if (!result) throw std::invalid_argument("Column Sparse Matrix cannot be loaded for training data file!");
	result = sparse::loadFileAsSparseMatrix(rowMatrixTrain, train_data_file);
	if (!result) throw std::invalid_argument("Row Sparse Matrix cannot be loaded for training data file!");


	// generate TEST output data files
	int m = rowMatrixTrain.rows();
	int n = rowMatrixTrain.cols();
	int nnzs_test = rowMatrixTest.nonZeros();
	int nnzs_train = rowMatrixTrain.nonZeros();

	parameters.m = m;
	parameters.n = n;
	parameters.nnz_test = nnzs_test;
	parameters.nnz_train = nnzs_train;

	//std::cout << "m = "<<m<<" , n = "<<n<<" , nnz_test = "<<nnzs_test<<" , nnz_train = "<<nnzs_train<< '\n';

	if (Multithreaded) {
		std::cout << "\nMultithreading ON.\n";
		typedef void (*callback_col)(CMatrix&, const std::string&);
		typedef void (*callback_row)(RMatrix&, const std::string&);
		callback_col colFunc;
		callback_row rowFunc;

		auto funcCol = [](callback_col pColFunc, CMatrix& matrix, const std::string filename) {
			pColFunc(matrix, filename);
		};
		auto funcRow = [](callback_row pRowFunc, RMatrix& matrix, const std::string filename) {
			pRowFunc(matrix, filename);
		};
		// Process Testing Data
		colFunc = &sparse::COL_ROW_VALUE::serialize_ROW_BIN;
		std::thread th1(funcCol, colFunc, std::ref(colMatrixTest), output_folder+"R_test_coo.row.bin");
		colFunc = &sparse::COL_ROW_VALUE::serialize_COL_BIN;
		std::thread th2(funcCol, colFunc, std::ref(colMatrixTest), output_folder+"R_test_coo.col.bin");
		colFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
		std::thread th3(funcCol, colFunc, std::ref(colMatrixTest), output_folder+"R_test_coo.data.bin");
		// Process Training Data
		colFunc = &sparse::COL_ROW_VALUE::serialize_ROW_BIN;
		std::thread th4(funcCol, colFunc, std::ref(colMatrixTrain), output_folder+"R_train_coo.row.bin");
		colFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
		std::thread th5(funcCol, colFunc, std::ref(colMatrixTrain), output_folder+"R_train_csc.data.bin");
		rowFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
		std::thread th6(funcRow, rowFunc, std::ref(rowMatrixTrain), output_folder+"R_train_csr.data.bin");
		colFunc = &sparse::COL_ROW_VALUE::serialize_INDPTR;
		std::thread th7(funcCol, colFunc, std::ref(colMatrixTrain), output_folder+"R_train_csc.indptr.bin");
		rowFunc = &sparse::COL_ROW_VALUE::serialize_INDPTR;
		std::thread th8(funcRow, rowFunc, std::ref(rowMatrixTrain), output_folder+"R_train_csr.indptr.bin");
		colFunc = &sparse::COL_ROW_VALUE::serialize_INDICES;
		std::thread th9(funcCol, colFunc, std::ref(colMatrixTrain), output_folder+"R_train_csc.indices.bin");
		rowFunc = &sparse::COL_ROW_VALUE::serialize_INDICES;
		std::thread th10(funcRow, rowFunc, std::ref(rowMatrixTrain), output_folder+"R_train_csr.indices.bin");

		th1.join();
		th2.join();
		th3.join();
		th4.join();
		th5.join();
		th6.join();
		th7.join();
		th8.join();
		th9.join();
		th10.join();
	}
	else {
		// Process Testing Data
		sparse::COL_ROW_VALUE::serialize_ROW_BIN(colMatrixTest, output_folder+"R_test_coo.row.bin");
		sparse::COL_ROW_VALUE::serialize_COL_BIN(colMatrixTest, output_folder+"R_test_coo.col.bin");
		sparse::COL_ROW_VALUE::serialize_DATA_BIN(colMatrixTest, output_folder+"R_test_coo.data.bin");
		// Process Training Data
		sparse::COL_ROW_VALUE::serialize_ROW_BIN(colMatrixTrain, output_folder+"R_train_coo.row.bin");
		sparse::COL_ROW_VALUE::serialize_DATA_BIN(colMatrixTrain, output_folder+"R_train_csc.data.bin");
		sparse::COL_ROW_VALUE::serialize_DATA_BIN(rowMatrixTrain, output_folder+"R_train_csr.data.bin");
		sparse::COL_ROW_VALUE::serialize_INDPTR(colMatrixTrain, output_folder+"R_train_csc.indptr.bin");
		sparse::COL_ROW_VALUE::serialize_INDPTR(rowMatrixTrain, output_folder+"R_train_csr.indptr.bin");
		sparse::COL_ROW_VALUE::serialize_INDICES(colMatrixTrain, output_folder+"R_train_csc.indices.bin");
		sparse::COL_ROW_VALUE::serialize_INDICES(rowMatrixTrain, output_folder+"R_train_csr.indices.bin");
	}
	printf("Input Data processed into binaries.\n");
}

int main(int argc, char **argv)
{
	if (argc != 4) {
		printf("Usage: training_data testing_data output_folder.\n");
		return 0;
	}
	std::string training_data_file(argv[1]);
	std::string test_data_file(argv[2]);
	std::string output_folder(argv[3]);
	std::string slash("/");
	if (!endsWith(output_folder, slash)) output_folder += "/";
	sparse::TestTrainDataHeader parameters;

	process_test_train_data(test_data_file, training_data_file, output_folder, parameters);


	cudaSetDevice(DEVICEID);
	int* csrRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (parameters.m + 1) * sizeof(csrRowIndexHostPtr[0])) );
	int* csrColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, parameters.nnz_train * sizeof(csrColIndexHostPtr[0])) );
	float* csrValHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, parameters.nnz_train * sizeof(csrValHostPtr[0])) );
	float* cscValHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscValHostPtr, parameters.nnz_train * sizeof(cscValHostPtr[0])) );
	int* cscRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr, parameters.nnz_train * sizeof(cscRowIndexHostPtr[0])) );
	int* cscColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr, (parameters.n+1) * sizeof(cscColIndexHostPtr[0])) );
	int* cooRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cooRowIndexHostPtr, parameters.nnz_train * sizeof(cooRowIndexHostPtr[0])) );

	//calculate X from thetaT first, need to initialize thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, parameters.n * parameters.F * sizeof(thetaTHost[0])) );

	float* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, parameters.m * parameters.F * sizeof(XTHost[0])) );

	//initialize thetaT on host
	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < parameters.n * parameters.F; k++)
		thetaTHost[k] = 0.2*((float) rand() / (float)RAND_MAX);
	//CG needs to initialize X as well
	for (int k = 0; k < parameters.m * parameters.F; k++)
		XTHost[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);;
	printf("*******start loading training and testing sets to host.\n");
	//testing set
	int* cooRowIndexTestHostPtr = (int *) malloc(
			parameters.nnz_test * sizeof(cooRowIndexTestHostPtr[0]));
	int* cooColIndexTestHostPtr = (int *) malloc(
			parameters.nnz_test * sizeof(cooColIndexTestHostPtr[0]));
	float* cooValHostTestPtr = (float *) malloc(parameters.nnz_test * sizeof(cooValHostTestPtr[0]));

	struct timeval tv0;
	gettimeofday(&tv0, NULL);

	/* load sparseMatrixBins */
	loadCooSparseMatrixBin( (output_folder + "R_test_coo.data.bin").c_str(), (output_folder + "R_test_coo.row.bin").c_str(),
							(output_folder + "R_test_coo.col.bin").c_str(),
							cooValHostTestPtr, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, parameters.nnz_test);

	loadCSRSparseMatrixBin( (output_folder + "R_train_csr.data.bin").c_str(), (output_folder + "R_train_csr.indptr.bin").c_str(),
							(output_folder + "R_train_csr.indices.bin").c_str(),
							csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr, parameters.m, parameters.nnz_train);

	loadCSCSparseMatrixBin( (output_folder + "R_train_csc.data.bin").c_str(), (output_folder + "R_train_csc.indices.bin").c_str(),
							(output_folder +"R_train_csc.indptr.bin").c_str(),
							cscValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, parameters.n, parameters.nnz_train);

	loadCooSparseMatrixRowPtrBin( (output_folder + "R_train_coo.row.bin").c_str(), cooRowIndexHostPtr, parameters.nnz_train);

#define DEBUG 1
#ifdef DEBUG
    printf("\nloaded training csr to host; print data, row and col array\n");
	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%.1f ", csrValHostPtr[i]);
	}
	printf("\n");

	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%d ", csrRowIndexHostPtr[i]);
	}
	printf("\n");
	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%d ", csrColIndexHostPtr[i]);
	}
	printf("\n");

	printf("\nloaded testing coo to host; print data, row and col array\n");
	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%.1f ", cooValHostTestPtr[i]);
	}
	printf("\n");

	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%d ", cooRowIndexTestHostPtr[i]);
	}
	printf("\n");
	for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
		printf("%d ", cooColIndexTestHostPtr[i]);
	}
	printf("\n");

#endif


	double t0 = seconds();
	doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
		cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
		cooRowIndexHostPtr, thetaTHost, XTHost,
		cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
		parameters.m, parameters.n, parameters.F, parameters.nnz_train, parameters.nnz_test, parameters.lambda,
		ITERS, parameters.X_BATCH, parameters.THETA_BATCH, DEVICEID);
	printf("\ndoALS takes seconds: %.3f for F = %d\n", seconds() - t0, parameters.F);


	cudaFreeHost(csrRowIndexHostPtr);
	cudaFreeHost(csrColIndexHostPtr);
	cudaFreeHost(csrValHostPtr);
	cudaFreeHost(cscValHostPtr);
	cudaFreeHost(cscRowIndexHostPtr);
	cudaFreeHost(cscColIndexHostPtr);
	cudaFreeHost(cooRowIndexHostPtr);
	cudaFreeHost(XTHost);
	cudaFreeHost(thetaTHost);
	cudacall(cudaDeviceReset());
	printf("\nALS Done.\n");

	return 0;
}
