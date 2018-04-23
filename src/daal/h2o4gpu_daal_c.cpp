/*
 * h2o4gpu_daal_c.cpp
 *
 *  Created on: Apr 2, 2018
 *      Author: monika
 */
#include <string>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <sstream>
#include "h2o4gpu_daal_c.h"
#include "iinput.h"
#include "svd.h"
#include "linear_regression.h"
#include "ridge_regression.h"

using namespace H2O4GPU::DAAL;

// Daal input
void* CreateDaalInput(double *pData, size_t m_dim, size_t n_dim) {
	std::cout << "prvni pdata: "<< pData[0] << ", " << pData[1] << std::endl;
	return new(std::nothrow) HomogenousDaalData<double>(pData, m_dim, n_dim);
}

void* CreateDaalInputFeaturesDependent(double* featuresData, size_t m_features, size_t n_features,
	double* dependentData, size_t m_dependent, size_t n_dependent) {
	return new(std::nothrow) HomogenousDaalData<double>(featuresData, m_features, n_features,
						dependentData, m_dependent, n_dependent);
}
void* GetFeaturesData(void* input) {
	auto fd = static_cast<IInput<double> *>(input);
	return const_cast<void *>(static_cast<const void *>(&fd->getFeaturesTable()));
}
void* GetDependentTable(void* input) {
	auto dt = static_cast<IInput<double> *>(input);
	return const_cast<void *>(static_cast<const void *>(&dt->getDependentTable()));
}

void* CreateDaalInputFile(const char* filename) {
	return new(std::nothrow) HomogenousDaalData<std::string>(filename);
}
void* CreateDaalInputFileFeaturesDependent(const char* filename, size_t features, size_t dependentVariables) {
	return new(std::nothrow) HomogenousDaalData<std::string>(filename, features, dependentVariables);
}
void DeleteDaalInput(void* input) {
	delete static_cast<IInput<double> *>(input);
}
void PrintDaalNumericTablePtr(void *input,const char* msg, size_t rows, size_t cols) {
	std::cout << "try to print\n";
	try {
		auto hdd = static_cast<HomogenousDaalData<double> *>(input);
		PrintTable pt;
		std::cout << "test pt\n";
		pt.print(hdd->getNumericTable(), std::string(msg), rows, cols);
	} CATCH_DAAL
}

void PrintNTP(void* input, const char* msg, size_t rows, size_t cols) {
	std::cout << "PrintNTP\n";
	try {
		NumericTablePtr* ntp = static_cast<NumericTablePtr *>(input);
		PrintTable pt;
		pt.print(*ntp, std::string(msg), rows, cols);
	} CATCH_DAAL
}

// Singular Value Decomposition
void* CreateDaalSVD(void* input) {
	try {
		IInput<double>* in = static_cast<IInput<double> *>(input);
		return new(std::nothrow) SVD(in);
	} CATCH_DAAL
	return nullptr;
}

void DeleteDaalSVD(void* input) {
	delete static_cast<SVD *>(input);
}
void FitDaalSVD(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		psvd->fit();
	} CATCH_DAAL
}
void* GetDaalSVDSigma(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return const_cast<void *>
			(static_cast<const void*>(&psvd->getSingularValues()));
	} CATCH_DAAL
	return nullptr;
}
const void* GetDaalRightSingularMatrix(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return static_cast<const void*>(&psvd->getRightSingularMatrix());
	} CATCH_DAAL
	return nullptr;
}
const void* GetDaalLeftSingularMatrix(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return static_cast<const void*>(&psvd->getLeftSingularMatrix());
	} CATCH_DAAL
	return nullptr;
}


// Regression
void* CreateDaalRidgeRegression(void* input) {
	try {
		auto in = static_cast<IInput<double> *>(input);
		return new(std::nothrow) RidgeRegression(in);
	} CATCH_DAAL
	return nullptr;
}
void DeleteDaalRidgeRegression(void* regression) {
	delete static_cast<RidgeRegression *>(regression);
}
void TrainDaalRidgeRegression(void *regression) {
	try {
		auto ridgeRegression = static_cast<RidgeRegression *>(regression);
		ridgeRegression->train();
	} CATCH_DAAL
}
void PredictDaalRidgeRegression(void* regression, void* input) {
	try {
		auto reg = static_cast<RidgeRegression* >(regression);
		auto in = static_cast<IInput<float> *>(input);
		std::cout << "predictdaalridgeregression ;\n";
		reg->predict(*in);
	} CATCH_DAAL
}
const void* GetDaalRidgeRegressionBeta(void* regression) {
	try {
		auto reg = static_cast<RidgeRegression *>(regression);

		return static_cast<const void* >(&reg->getBeta());
		//auto beta = static_cast<const void*>(&reg->getBeta());

		//typedef std::remove_cv<decltype(beta)>::type unconst_type;
		//return const_cast<void *>(beta);
	} CATCH_DAAL
	return nullptr;
}
void* GetDaalRidgeRegressionPredictionData(void* regression) {
	try {
		auto reg = static_cast<RidgeRegression *>(regression);
		return const_cast<void *>(
				static_cast<const void*>(&reg->getPredictionData()));
	} CATCH_DAAL
	return nullptr;
}

// Linear Regression
void* CreateDaalLinearRegression(void* input) {
	try {
		IInput<float>* in = static_cast<IInput<float> *>(input);
		return new(std::nothrow) LinearRegression(*in);
	} CATCH_DAAL
	return nullptr;
}
void DeleteDaalLinearRegression(void* regression) {
	delete static_cast<LinearRegression *>(regression);
}
void TrainDaalLinearRegression(void *regression) {
	try {
		auto reg = static_cast<LinearRegression *>(regression);
		reg->train();
	} CATCH_DAAL
}
void PredictDaalLinearRegression(void* regression, void* input) {
	try {
		auto reg = static_cast<LinearRegression *>(regression);
		auto in = static_cast<IInput<float> *>(input);
		reg->predict(*in);
	} CATCH_DAAL
}
const void* GetDaalLinearRegressionBeta(void* regression) {
	try {
		auto reg = static_cast<LinearRegression *>(regression);
		return &reg->getBeta();
	} CATCH_DAAL
	return nullptr;
}
const void* GetDaalLinearRegressionPredictionData(void* regression) {
	try {
		auto reg = static_cast<LinearRegression *>(regression);
		return &reg->getPredictionData();
	} CATCH_DAAL
	return nullptr;
}
