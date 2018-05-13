/*
 * eigen_sparse_manager.h
 *
 *  Created on: May 7, 2018
 *      Author: monika
 */

#ifndef EIGEN_SPARSE_MANAGER_H_
#define EIGEN_SPARSE_MANAGER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <cstddef>
#include <type_traits>

#if DEBUG
#define log(...) {											\
	char str[82];											\
	sprintf(str, __VA_ARGS__);								\
	std::cout << "<" << __FILE__ << "><"<<__FUNCTION__<<	\
			<<"><L:"<< __LINE__ << ">: "<< str << std::endl;\
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
		int f;
		int X_BATCH; 		// for smaller RAM increase X_BATCH
		int THETA_BATCH; 	// for smaller RAM increase THETA_BATCH
		void setF(int F) {
			// must be modulus of 10
			if (F%10 != 0)
				throw std::invalid_argument("F must be modulus of 10");
			this->f = F;
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
	void split_path(const std::string& path, std::string& basedir, std::string& filename) {
		std::size_t dirPos = path.find_last_of("/");
		basedir = path.substr(0, dirPos);
		filename = path.substr(dirPos+1, path.length());
	}

	template<typename Type>
	inline bool getLine(std::stringstream& line, int& m, int& n, int& i, int& j, Type& value)
	{
		// row, col, val
		line >> i >> j >> value;
		i--;j--;
		return (i >= 0 && j >= 0 && i < m && j < n) ? true : false;
	}

	/* reads file into matrix */
	template<typename SparseMatrixType>
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
				line >> m >> n >> nnz;
				if (m > 0 && n > 0 && nnz > 0) {
					readSize = true;
					log("size: "<<m<<","<<n<<","<<nnz<<"\n");
					matrix.resize(m,n);
					matrix.reserve(nnz);
				}
			} else {
				int i(-1), j(-1);
				Type value;
				if (getLine(line, m, n, i, j, value)) {
					++count;
					elements.push_back(T(i,j,value));
				} else {
					std::cerr << "Invalid read: "<< i << ","<< j << "\n";
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

	/* by default optoins = colMajor, all test data must be col_major
	 * training data in col_major and row_major
	 */
	template <typename Type, int options, typename StorageIndex>
	void serialize_rows_cols_data(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string&& filename) {
		typedef typename Eigen::SparseMatrix<Type, options, StorageIndex>::InnerIterator InnerIterator;
		std::fstream wFile;
		StorageIndex nnzs = matrix.nonZeros();
		StorageIndex rows = matrix.rows();
		StorageIndex cols = matrix.cols();

		std::cout << "serialize rows cols data\n";

		std::string fileName, dirName;
		split_path(filename, dirName, fileName);
		std::string newPath = dirName + '/' + fileName;
		std::cout << newPath << std::endl;

		wFile.open(newPath+".row.bin", std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)&(rows), sizeof(StorageIndex));
			wFile.write((const char *)&(cols), sizeof(StorageIndex));
			wFile.write((const char *)&(nnzs), sizeof(StorageIndex));
			wFile.write((const char *)(matrix.innerIndexPtr()), sizeof(StorageIndex) * nnzs);
			wFile.close();
		}
		wFile.open(newPath+".data.bin", std::ios::binary | std::ios::out);
		if (wFile.is_open()) {
			wFile.write((const char *)&(rows), sizeof(StorageIndex));
			wFile.write((const char *)&(cols), sizeof(StorageIndex));
			wFile.write((const char *)&(nnzs), sizeof(StorageIndex));
			wFile.write((const char *)(matrix.valuePtr()), sizeof(Type) * nnzs);
			wFile.close();
		}
		wFile.open(newPath+".col.bin", std::ios::binary | std::ios::out);
		StorageIndex *pcols = new StorageIndex[nnzs];
		for (size_t k = 0; k < matrix.outerSize(); ++k) {
			size_t index = 0;
			for (InnerIterator it(matrix, k); it; ++it)
			{
				pcols[index++]=it.col();
			}
		}
		if (wFile.is_open()) {
			wFile.write((const char *)&(rows), sizeof(StorageIndex));
			wFile.write((const char *)&(cols), sizeof(StorageIndex));
			wFile.write((const char *)&(nnzs), sizeof(StorageIndex));
			wFile.write((const char *)pcols, sizeof(Type)*nnzs);
			wFile.close();
		}
	}

	// for testing
	template <typename Type, int options, typename StorageIndex>
		void deserialize_test_data(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
			std::fstream wFile;
			StorageIndex nnzs;
			StorageIndex * innerPtr = new StorageIndex[10];
			wFile.open(filename+".row.bin", std::ios::binary | std::ios::in);
			if (wFile.is_open()) {
				wFile.read((char *)&nnzs, sizeof(StorageIndex));
				wFile.read((char *)(innerPtr), sizeof(Type) * nnzs);
				for (int i = 0; i < nnzs; i++) {
					std::cout << innerPtr[i] << " - ";
				}
				wFile.close();
			}
		}


	// Serialization - deserialization to binaries
	template <typename Type, int options, typename StorageIndex>
	void serialize(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
		typedef typename Eigen::Triplet<Type> Triplet;
		std::vector<Triplet> result;
		int nnz = matrix.nonZeros();
		matrix.makeCompressed();

		std::fstream writeFile;
		writeFile.open(filename, std::ios::binary | std::ios::out);
	}



} // namespase sparse



#endif /* EIGEN_SPARSE_MANAGER_H_ */
