/*
 * sparse_manager.h
 *
 *  Created on: Jun 17, 2018
 *      Author: monika
 */

#ifndef SPARSE_MANAGER_H_
#define SPARSE_MANAGER_H_

#include "helper/debug_output.h"

#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <cstddef>
#include <type_traits>
#include <stdexcept>



namespace matrix_factorization {

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
			line >> m >> m >> nnz;
			if (m > 0 && n > 0 && nnz > 0) {
				readSize = true;
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

/*
template<typename Type, typename StorageIndex=Type>
class Serialize {
public:

};
*/


} // namespace matrix_factorization



#endif /* SPARSE_MANAGER_H_ */
