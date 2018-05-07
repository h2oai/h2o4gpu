/*
 * eigen_sparse_matrix.h
 *
 *  Created on: May 7, 2018
 *      Author: monika
 */

#ifndef SRC_MXFACTORIZATION_EIGEN_SPARSE_MATRIX_H_
#define SRC_MXFACTORIZATION_EIGEN_SPARSE_MATRIX_H_

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
#include <vector>

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

	template<typename Type>
	inline bool getLine(const std::stringstream& line, int& m, int& n, int& i, int& j, Type& value)
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
		typedef Eigen::Triplet<Scalar, int> T;
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
				if (GetLine(line, m, n, i, j, value)) {
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

	// Serialization - deserialization to binaries
	template <typename Type, int options, typename StorageIndex>
	void serialize(Eigen::SparseMatrix<Type, options, StorageIndex>& matrix, const std::string& filename) {
		typedef typename Eigen::Triplet<Type> Triplet;
		std::vector<Triplet> result;
		int nnz = matrix.nonZeros();
		matrix.makeCompressed();

		std::fstream writeFile;
		writeFile.open(filename, ios::binary | ios::out);
	}



} // namespase sparse



#endif /* EIGEN_SPARSE_MANAGER_H_ */




#endif /* SRC_MXFACTORIZATION_EIGEN_SPARSE_MATRIX_H_ */
