/*
 * eigen_sparse_manager.h
 *
 *  Created on: May 7, 2018
 *      Author: monika
 *
 *  output:
 *  (1. Traning data)
 *  (1.1 csr.data.bin) = R_test_coo.data.bin (OK) = serialize_test_data_coo_data_bin
 *  (1.2 csr.indices.bin) = R_train_csr.indices.bin (OK) = serialize_indices
 *  (1.3 csr.indptr.bin) -> R_train_csr.indptr.bin (OK) = serialize_indPtr
 *  (1.4 csc.data.bin) = R_test_coo.data.bin (OK) = serialize_test_data_coo_data_bin (with <ColMajor>)
 *  (1.5 csc.indices.bin) = csc (OK)
 *  (1.6 csc.indptr.bin) = R_train_csc.indptr.bin (OK) = serialize_indPtr
 *  (2. Test data)
 *  (2.1 csr.data.bin) -> R_test_coo.data.bin (OK) = serialize_test_data_coo_data_bin
 *  (2.2 csr.row.bin) -> R_test_coo.row.bin (OK) = serialize_test_data_coo_row_bin
 *  (2.3 coo.col.bin) -> R_test_coo.col.bin (OK) = serialize_test_data_coo_col_bin
 *
 * python script:
 * train_data_file = '/home/monika/h2o/src/cuda_cu/cuMF/cumf_als/data/netflix/reverse_netflix.mm'
 * test_data_file = '/home/monika/h2o/src/cuda_cu/cuMF/cumf_als/data/netflix/reverse_netflix.mme'
 */

#ifndef EIGEN_SPARSE_MANAGER_H_
#define EIGEN_SPARSE_MANAGER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>
//#include <Eigen/Dense>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <stdexcept>

#if DEBUG
#define log(...) {											\
	char str[82];											\
	sprintf(str, __VA_ARGS__);								\
	std::cout << "LOG: <" << __FILE__ << "><"<<__FUNCTION__<< "> "<<str << std::endl;\
}
#else
#define log(...)
#endif


namespace sparse {

	struct TestTrainDataHeader {
		int m;
		int n;
		int nnz_test;
		int nnz_train;
		float lambda;
		int F;
		int X_BATCH; 		// for smaller RAM increase X_BATCH
		int THETA_BATCH; 	// for smaller RAM increase THETA_BATCH
		void setF(int F) {
			// must be modulus of 10
			if (F%10 != 0)
				throw std::invalid_argument("F must be modulus of 10");
			this->F = F;
		}
		void setLambda(float L) {
			this->lambda = L;
		}
		void setXBatch(int x_b) {
			this->X_BATCH = x_b;
		}
		void setThetaBatch(int t_b) {
			this->THETA_BATCH = t_b;
		}

	};


	template<typename Type>
	inline bool getLine(std::stringstream& line, int& m, int& n, int& i, int& j, Type& value)
	{
		// row, col, val
		line >> i >> j >> value;
		i--;j--;
		return (i >= 0 && j >= 0 && i < m && j < n) ? true : false;
	}

	void split_path(const std::string& path, std::string& basedir, std::string& filename) {
		std::size_t dirPos = path.find_last_of("/");
		basedir = path.substr(0, dirPos);
		filename = path.substr(dirPos+1, path.length());
	}

	enum InputFormatFile { row_column_value, column_row_value };

	/* reads file into matrix */
	template<typename SparseMatrixType, InputFormatFile FormatFile = column_row_value>
	bool loadFileAsSparseMatrix(SparseMatrixType& matrix, const std::string& filename) {
		typedef typename SparseMatrixType::Scalar Type;
		std::ifstream input(filename.c_str(), std::ios::in);
		if (!input) return false;

		const int maxBufferSize = 2048;
		char buffer[maxBufferSize];
		bool readSize = false;
		typedef Eigen::Triplet<Type, int> T;
		std::vector<T> elements;

		int m(-1), n(-1), nnz(-1);
		int count = 0;
		while (input.getline(buffer, maxBufferSize)) {
			// skip comments as % or #
			if (buffer[0] == '%' || buffer[0] == '#') continue;
			std::stringstream line(buffer);
			if (!readSize) {
				// first line m, n, nnz
				if (FormatFile == column_row_value)
					line >> n >> m >> nnz;
				else
					line >> m >> n >> nnz;
				if (m > 0 && n > 0 && nnz > 0) {
					readSize = true;
					matrix.resize(m,n);
					matrix.reserve(nnz);
				}
			} else {
				int i(-1), j(-1);
				Type value;
				if (FormatFile == column_row_value) {
					if (getLine(line, n, m, j, i, value)) {
						++count;
						elements.push_back(T(i,j,value));
					} else {
						std::cerr << "Invalid read: "<< i << ","<< j << "\n";
					}
				} else {
					if (getLine(line, m, n, i, j, value)) {
						++count;
						elements.push_back(T(i,j,value));
					} else {
						std::cerr << "Invalid read: "<< i << ","<< j << "\n";
					}
				}
			}
		}
		matrix.setFromTriplets(elements.begin(), elements.end());
		if (count != nnz)
			std::cerr << count << "!=" << nnz << "\n";
		input.close();
		return true;
	}

	template <typename Type, int options, typename StorageIndex>
	int getNNZ(const Eigen::SparseMatrix<Type, options, StorageIndex>& matrix) {
		return matrix.nonZeros();
	}

	template<typename Type, int options, typename StorageIndex>
	void serialize_indices(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
		std::fstream wFile;
		StorageIndex nnzs = matrix.nonZeros();

		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)(matrix.innerIndexPtr()), sizeof(StorageIndex) * nnzs);
			wFile.close();
		}

#if DEBUG
		std::cout << "indices\n";
		auto iip = matrix.innerIndexPtr();
		for (StorageIndex i = 0; i < nnzs; i++) {
			std::cout << iip[i] << ' ';
		}
		std::cout << std::endl;
#endif
	}

	namespace COL_ROW_VALUE {

	/* row.bin */
	template <typename Type, typename StorageIndex=int>
	void serialize_ROW_BIN(Eigen::SparseMatrix<Type, Eigen::ColMajor, StorageIndex>& matrix, const std::string& filename="R_test_coo.col.bin") {
		typedef typename Eigen::SparseMatrix<Type, Eigen::ColMajor, StorageIndex>::InnerIterator InnerIterator;
		std::fstream wFile;
		StorageIndex nnzs = matrix.nonZeros();

		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {

			wFile.write((const char *)(matrix.innerIndexPtr()), sizeof(StorageIndex) * nnzs);
			wFile.close();
		}
#if DEBUG
		log(" ");
		auto iip = matrix.innerIndexPtr();
		for (int i = 0; i < nnzs; i++) {
			std::cout << iip[i]<< ' ';
			if (i > 10) break;
		}
		std::cout << std::endl;
#endif
	}

	/* col.bin */
	template <typename Type, typename StorageIndex=Type>
	void serialize_COL_BIN(Eigen::SparseMatrix<Type, Eigen::ColMajor, StorageIndex>& matrix, const std::string& filename="R_test_coo.row.bin") {
		typedef typename Eigen::SparseMatrix<Type, Eigen::ColMajor, StorageIndex>::InnerIterator InnerIterator;
		std::fstream wFile;

		StorageIndex rows = matrix.rows();
		StorageIndex cols = matrix.cols();
		StorageIndex nnzs = matrix.nonZeros();
		StorageIndex outS = matrix.outerSize();

		std::vector<StorageIndex> outer_rows;
		for (size_t k = 0; k < outS; ++k) {
			for(InnerIterator it(matrix, k); it; ++it) {
				outer_rows.push_back(it.outer());
			}
		}

#if DEBUG
		log(" ");
		for (size_t i = 0; i < outer_rows.size(); ++i) {
			std::cout << outer_rows[i] << ' ';
			if (i > 10) break;
		}
		std::cout << std::endl;
#endif

		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)&outer_rows[0], outer_rows.size() * sizeof(StorageIndex));
			wFile.close();
		}
	}

	/* col.data */
	template <typename Type, int options, typename StorageIndex=Type>
	void serialize_DATA_BIN(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename="R_test_coo.data.bin") {
		typedef typename Eigen::SparseMatrix<Type, options, StorageIndex>::InnerIterator InnerIterator;
		std::fstream wFile;

		StorageIndex rows = matrix.rows();
		StorageIndex cols = matrix.cols();
		StorageIndex nnzs = matrix.nonZeros();

		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)(matrix.valuePtr()), sizeof(StorageIndex) * nnzs);
			wFile.close();
		}

#if DEBUG
		log(" ");
		auto iip = matrix.valuePtr();
		for (int i = 0; i < nnzs; i++) {
			std::cout << iip[i]<< ' ';
			if (i > 10) break;
		}
		std::cout << std::endl;
#endif
	}

	/* train Indptr */
	template <typename Type, int options, typename StorageIndex>
	std::vector<StorageIndex> indPtr(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix) {
		std::vector<StorageIndex> indPtrVec = {0}; // initialize with zero
		typedef typename Eigen::SparseMatrix<Type, options>::InnerIterator iit;
		for (size_t k = 0; k < matrix.outerSize(); ++k) {
			size_t index = 0;
			for (iit it(matrix, k); it; ++it) {
				++index;
			}
			StorageIndex pitch = indPtrVec.back();
			indPtrVec.push_back(pitch+index);
		}
		return indPtrVec;
		if (indPtrVec.size() != matrix.rows()+1)
			throw std::length_error("indPtr has not the correct size: "+ std::to_string(matrix.rows()+1));
	}

	template <typename Type, int options, typename StorageIndex>
	void serialize_INDPTR(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
		std::fstream wFile;
		StorageIndex nnzs = matrix.nonZeros();
		StorageIndex rows = matrix.rows();
		StorageIndex cols = matrix.cols();

		auto indPtrVec = indPtr(matrix);
		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)(indPtrVec.data()), sizeof(StorageIndex) * indPtrVec.size());
			wFile.close();
		}

#if DEBUG
		log(" ");
		int counter = 0;
		for (typename std::vector<StorageIndex>::const_iterator i = indPtrVec.begin(); i != indPtrVec.end(); ++i, ++counter) {
			std::cout << *i << ' ';
			if (counter > 10) break;
		}
		std::cout << std::endl;
#endif
	}

	/* Indices */
	template<typename Type, int options, typename StorageIndex>
	void serialize_INDICES(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
		std::fstream wFile;
		StorageIndex nnzs = matrix.nonZeros();

		wFile.open(filename, std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)(matrix.innerIndexPtr()), sizeof(StorageIndex) * nnzs);
			wFile.close();
		}
#if DEBUG
		log(" ");
		auto iip = matrix.innerIndexPtr();
		for (StorageIndex i = 0; i < nnzs; i++) {
			std::cout << iip[i] << ' ';
			if (i > 10) break;
		}
		std::cout << std::endl;
#endif
	}

	} // end of namespace COL_ROW_VALUE

} // namespase sparse



#endif /* EIGEN_SPARSE_MANAGER_H_ */
