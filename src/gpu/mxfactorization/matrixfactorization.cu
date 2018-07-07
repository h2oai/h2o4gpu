/*************************************************************************
 *
 * Copyright (c) 2018, H2O.ai, Inc. All rights reserved.
 *
 ************************************************************************/

#include "../../include/solver/matrixfactorization.h"
#include "../../common/logger.h"
#include "als.h"
#include "eigen_sparse_manager.h"
#include "host_utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <thread>

namespace matrixfactorization {

	/**
	 * Input data as two data sources, training data sets and testing data sets,
	 * data is read in memory for calculation
	 * 
	 * @param test_data_file
	 * @param train_data_file
	 * @param parameters
	 */
	
	template<typename Type, bool Multithreaded>
	void process_data(const char* test_data_file,
			const char* train_data_file,
			const char* output_folder,
			sparse::TestTrainDataHeader& parameters)
	{

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

		log_info(sparse::Verbose::value, "Sparse matrix %dx%d, nnz_test=%d, nnz_train=%d", m, n, nnzs_test, nnzs_train);

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
			std::thread th1(funcCol, colFunc, std::ref(colMatrixTest), sparse::file_name_creator(output_folder,"R_test_coo.row.bin"));
			colFunc = &sparse::COL_ROW_VALUE::serialize_COL_BIN;
			std::thread th2(funcCol, colFunc, std::ref(colMatrixTest), sparse::file_name_creator(output_folder,"R_test_coo.col.bin"));
			colFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
			std::thread th3(funcCol, colFunc, std::ref(colMatrixTest), sparse::file_name_creator(output_folder,"R_test_coo.data.bin"));
			// Process Training Data
			colFunc = &sparse::COL_ROW_VALUE::serialize_ROW_BIN;
			std::thread th4(funcCol, colFunc, std::ref(colMatrixTrain), sparse::file_name_creator(output_folder,"R_train_coo.row.bin"));
			colFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
			std::thread th5(funcCol, colFunc, std::ref(colMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csc.data.bin"));
			rowFunc = &sparse::COL_ROW_VALUE::serialize_DATA_BIN;
			std::thread th6(funcRow, rowFunc, std::ref(rowMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csr.data.bin"));
			colFunc = &sparse::COL_ROW_VALUE::serialize_INDPTR;
			std::thread th7(funcCol, colFunc, std::ref(colMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csc.indptr.bin"));
			rowFunc = &sparse::COL_ROW_VALUE::serialize_INDPTR;
			std::thread th8(funcRow, rowFunc, std::ref(rowMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csr.indptr.bin"));
			colFunc = &sparse::COL_ROW_VALUE::serialize_INDICES;
			std::thread th9(funcCol, colFunc, std::ref(colMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csc.indices.bin"));
			rowFunc = &sparse::COL_ROW_VALUE::serialize_INDICES;
			std::thread th10(funcRow, rowFunc, std::ref(rowMatrixTrain), sparse::file_name_creator(output_folder,"R_train_csr.indices.bin"));

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
			sparse::COL_ROW_VALUE::serialize_ROW_BIN(colMatrixTest, sparse::file_name_creator(output_folder, "R_test_coo.row.bin"));
			sparse::COL_ROW_VALUE::serialize_COL_BIN(colMatrixTest, sparse::file_name_creator(output_folder,"R_test_coo.col.bin"));
			sparse::COL_ROW_VALUE::serialize_DATA_BIN(colMatrixTest, sparse::file_name_creator(output_folder,"R_test_coo.data.bin"));
			// Process Training Data
			sparse::COL_ROW_VALUE::serialize_ROW_BIN(colMatrixTrain, sparse::file_name_creator(output_folder,"R_train_coo.row.bin"));
			sparse::COL_ROW_VALUE::serialize_DATA_BIN(colMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csc.data.bin"));
			sparse::COL_ROW_VALUE::serialize_DATA_BIN(rowMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csr.data.bin"));
			sparse::COL_ROW_VALUE::serialize_INDPTR(colMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csc.indptr.bin"));
			sparse::COL_ROW_VALUE::serialize_INDPTR(rowMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csr.indptr.bin"));
			sparse::COL_ROW_VALUE::serialize_INDICES(colMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csc.indices.bin"));
			sparse::COL_ROW_VALUE::serialize_INDICES(rowMatrixTrain, sparse::file_name_creator(output_folder,"R_train_csr.indices.bin"));
		}
		log_info(sparse::Verbose::value, "Input Data processed into binaries.");
	}
	
	/** 
	 * Conduct Matrix Factorization on Eigen::Matrix with float type
	 * 
	 * @param test_data_file
	 * @param train_data_file
	 * @param thetaTHost
	 * @param XTHost
	 * @param F
	 * @param lambda
	 * @param X_BATCH
	 * @param THETA_BATCH 
	 */
	void matrixfactorization_float(const char* test_data_file,
			const char* train_data_file,
			const char* output_folder,
			float *thetaTHost,
			float* XTHost,
			int F,
			float lambda,
			int X_BATCH,
			int THETA_BATCH,
			int n_iter,
			int verbose,
			int gpu_id) {
					
		sparse::Verbose::value = verbose;
		sparse::TestTrainDataHeader parameters;
		parameters.setF(F);
		parameters.setLambda(lambda);
		parameters.setXBatch(X_BATCH);
		parameters.setThetaBatch(THETA_BATCH);

		process_data<float, true>(test_data_file, train_data_file, output_folder, parameters);

		cudaSetDevice(gpu_id);
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

		//calculatethetaTHost, XTHost X from thetaT first, need to initialize thetaT
		//float* thetaTHost;
		cudacall(cudaMallocHost( (void** ) &thetaTHost, parameters.n * parameters.F * sizeof(thetaTHost[0])) );

		//float* XTHost;
		cudacall(cudaMallocHost( (void** ) &XTHost, parameters.m * parameters.F * sizeof(XTHost[0])) );

		//initialize thetaT on host
		unsigned int seed = 0;
		srand (seed);
		for (int k = 0; k < parameters.n * parameters.F; k++)
			thetaTHost[k] = 0.2*((float) rand() / (float)RAND_MAX);
		//CG needs to initialize X as well
		for (int k = 0; k < parameters.m * parameters.F; k++)
			XTHost[k] = 0;//0.1*((float) rand() / (float)RAND_MAX);;
		log_info(sparse::Verbose::value, "Start loading training and testing sets to host.");
		//testing set
		int* cooRowIndexTestHostPtr = (int *) malloc(
				parameters.nnz_test * sizeof(cooRowIndexTestHostPtr[0]));
		int* cooColIndexTestHostPtr = (int *) malloc(
				parameters.nnz_test * sizeof(cooColIndexTestHostPtr[0]));
		float* cooValHostTestPtr = (float *) malloc(parameters.nnz_test * sizeof(cooValHostTestPtr[0]));

		struct timeval tv0;
		gettimeofday(&tv0, NULL);

		/* load sparseMatrixBins */
		loadCooSparseMatrixBin( sparse::file_name_creator(output_folder,"R_test_coo.data.bin"), 
								sparse::file_name_creator(output_folder,"R_test_coo.row.bin"),
								sparse::file_name_creator(output_folder,"R_test_coo.col.bin"),
								cooValHostTestPtr, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, parameters.nnz_test);

		loadCSRSparseMatrixBin( sparse::file_name_creator(output_folder,"R_train_csr.data.bin"), 
								sparse::file_name_creator(output_folder,"R_train_csr.indptr.bin"),
								sparse::file_name_creator(output_folder,"R_train_csr.indices.bin"),
								csrValHostPtr, csrRowIndexHostPtr, csrColIndexHostPtr, parameters.m, parameters.nnz_train);

		loadCSCSparseMatrixBin( sparse::file_name_creator(output_folder,"R_train_csc.data.bin"), 
								sparse::file_name_creator(output_folder,"R_train_csc.indices.bin"),
								sparse::file_name_creator(output_folder,"R_train_csc.indptr.bin"),
								cscValHostPtr, cscRowIndexHostPtr, cscColIndexHostPtr, parameters.n, parameters.nnz_train);

		loadCooSparseMatrixRowPtrBin( sparse::file_name_creator(output_folder,"R_train_coo.row.bin"), 
								cooRowIndexHostPtr, 
								parameters.nnz_train);


		log_debug(sparse::Verbose::value, "Loaded training csr to host; print data, row and col array.");
		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%.1f ", csrValHostPtr[i]);
		}

		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%d ", csrRowIndexHostPtr[i]);
		}
		
		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%d ", csrColIndexHostPtr[i]);
		}
		
		log_debug(sparse::Verbose::value, "Loaded training coo to host; print data, row and col array.");
		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%.1f ", cooValHostTestPtr[i]);
		}

		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%d ", cooRowIndexTestHostPtr[i]);
		}
		
		for (int i = 0; i < parameters.nnz_train && i < 10; i++) {
			log_debug(sparse::Verbose::value, "%d ", cooColIndexTestHostPtr[i]);
		}


		double t0 = seconds();
		doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
			cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
			cooRowIndexHostPtr, thetaTHost, XTHost,
			cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
			parameters.m, parameters.n, parameters.F, parameters.nnz_train, parameters.nnz_test, parameters.lambda,
			n_iter, parameters.X_BATCH, parameters.THETA_BATCH, gpu_id);
		log_info(sparse::Verbose::value, "doALS takes seconds: %.3f for F = %d\n", seconds() - t0, parameters.F);

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
	}
	
}  // namespace matrixfactorization

