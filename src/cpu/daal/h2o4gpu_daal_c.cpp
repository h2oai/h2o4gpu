/*
 * h2o4gpu_daal_c.cpp
 *
 *  Created on: Apr 2, 2018
 *      Author: monika
 */

#include "h2o4gpu_daal_c.h"
#include "iinput.hpp"
#include "regression.hpp"


extern "C" {
	// Daal input
	void* CreateDaalInput(double *pData, size_t m_dim, size_t n_dim) {
		return new(std::nothrow) HomogenousDaalData<double>(pData, m_dim, n_dim);
	}
	void* CreateDaalInput(double* featuresData, size_t m_features, size_t n_features,
							double* dependentData, size_t m_dependent, size_t n_dependent) {
		return new(std::nothrow) HomogenousDaalData(featuersData, m_features, n_features,
							dependentData, m_dependent, n_dependent);
	}
	void* CreateDaalInput(const char* filename) {
		return new(std::nothrow) HomogenousDaalData<std::string>(filename);
	}
	void* CreateDaalInput(const char* filename, size_t features, size_t dependentVariables) {
		return new(std;:nothrow) HomogenousDaalData<std::string>(filename, features, dependentVariables);
	}
	void DeleteDaalInput_d(void* input) {
		DeleteDaalInput<double>(input);
	}
	void printDaalInput_d(void *input,c onst char* msg) {
		printDaalInput<double>(input, msg);
	}
	void *getDaalNT_d(const void* input) {
		return getDaalNT<double>(input);
	}
	void printDaalNT(const void* input, const char* msg, size_t rows=0, size_t columns=0) {
		try {
			auto p = static_cast<const NumericTable *>(input);
			printNumericTable(p, msg, rows, columns);
		}
	}

	// Singular Value Decomposition
	void* CreateDaalSVD(const void* input) {
		try {
			auto in = static_cast<const IInput*>(input);
		} CATCH
		return new(std::nothrow) SVD(*in);
	}
	void* CreateDaalSVD(double* input, size_t m_dim, size_t n_dim) {
		return CreateSVD(input, m_dim, n_dim);
	}
	void DeleteSVD(void* pSVD) {
		delete static_cast<SVD *>(pSVD);
	}
	const void fitDaalSVD(const void* svd) {
		try {
			auto psvd = static_cast<const SVD* >(svd);
			psvd->fit();
		} CATCH
	}
	const void* getDaalSigmas(const void* svd) const {
		try {
			auto psvd = static_cast<const SVD* >(svd);
			return psvd->getSingularValues();
		} CATCH
		return nullptr;
	}
	const void* getDaalRightSingularMatrix(const void* svd) const {
		try {
			auto psvd = static_cast<const SVD* >(svd);
			return psvd->getRightSingularMatrix();
		} CATCH
		return nullptr;
	}
	const void* getDaalLeftSingularMatrix(const void* svd) const {
		try {
			auto psvd = static_cast<const SVD* >(svd);
			return psvd->getLeftSingularMatrix();
		} CATCH
		return nullptr;
	}
	// Regression
	void* CreateDaalRegression(const void* input) {
		try {
			auto in = static_cast<const IInput*>(input);
		} CATCH
		return new(std::nothrow) Regression(*in);
	}
	void* CreateDaalRegression() {
		return new(std::nothrow) Regression();
	}
	void DeleteDaalRegression(void* r) {
		delete static_cast<Regression *>(r);
	}
	const void* train(Regression::Type type, const void* regression) {
		try {
			Regression* reg= nullptr;
			if (type == Regression::type.linear)
				reg = dynamic_cast<LinearRegression *>(regression);
			else if(type == Regression::type.rigid)
				reg = dynamic_cast<RidgeRegression *>(regression);
			else
				throw "Either linear or rigid regression is possible.";
			return reg->train();
		} CATCH
		return nullptr;
	}
	const void* predict(Regression::Type type, const void* regression, const void* input) {
		try {
			Regression* reg = nullptr;
			auto in = static_cast<const IInput *>(input);
			if (type == Regression::type.linear)
				reg = dynamic_cast<LinearRegression *>(regression);
			else if (type == Regression::type.rigid)
				reg = dynamic_cast<RigidRegression *>(regression);
			else throw "Either linear or rigid regression is possible.";
			return reg->predict(*in);
		} CATCH
		return nullptr;
	}
	void setInput(Regression::Type type, const void* regression, const void* input) {
		try {
			auto in = static_cast<const IInput *>(input);
			Regression* reg = nullptr;
			if (type == Regression::type.linear)
				reg = dynamic_cast<LinearRegression *>(regression);
			else if (type == Regression::type.rigid)
				reg = dynamic_cast<RigidRegression *>(regression);
			else throw "Either linear or rigid regression is possible.";
			return reg->setInput(*in);
		}
	}
}

