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
#include "ridge_regression.h"
#include "linear_regression.h"

using namespace H2O4GPU::DAAL;

	// Daal input
void* CreateDaalInput(float *pData, size_t m_dim, size_t n_dim) {
	return new(std::nothrow) HomogenousDaalData<float>(pData, m_dim, n_dim);
}

void* CreateDaalInputFeaturesDependent(float* featuresData, size_t m_features, size_t n_features,
						float* dependentData, size_t m_dependent, size_t n_dependent) {
	return new(std::nothrow) HomogenousDaalData<float>(featuresData, m_features, n_features,
						dependentData, m_dependent, n_dependent);
}
void* CreateDaalInputFile(const char* filename) {
	return new(std::nothrow) HomogenousDaalData<std::string>(filename);
}
void* CreateDaalInputFileFeaturesDependent(const char* filename, size_t features, size_t dependentVariables) {
	return new(std::nothrow) HomogenousDaalData<std::string>(filename, features, dependentVariables);
}
void DeleteDaalInput(void* input) {
	delete static_cast<IInput<float> *>(input);
}
void PrintDaalNumericTablePtr(float *input,const char* msg, size_t rows, size_t cols) {
	try {
		const NumericTablePtr* input = static_cast<const NumericTablePtr *>(input);
		PrintTable pt;
		pt.print(*input, std::string(msg), rows, cols);
	} CATCH_DAAL
}
// Singular Value Decomposition
void* CreateDaalSVD(void* input) {
	try {
		IInput<float>* in = static_cast<IInput<float> *>(input);
		return new(std::nothrow) SVD(*in);
	} CATCH_DAAL
	return nullptr;
}
void DeleteDaalSVD(void* input) {
	delete static_cast<SVD *>(input);
}
void fitDaalSVD(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		psvd->fit();
	} CATCH_DAAL
}
const void* getDaalSVDSigma(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return static_cast<const void*>(&psvd->getSingularValues());
	} CATCH_DAAL
	return nullptr;
}
const void* getDaalRightSingularMatrix(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return static_cast<const void*>(&psvd->getRightSingularMatrix());
	} CATCH_DAAL
	return nullptr;
}
const void* getDaalLeftSingularMatrix(void* svd) {
	try {
		auto psvd = static_cast<SVD* >(svd);
		return static_cast<const void*>(&psvd->getLeftSingularMatrix());
	} CATCH_DAAL
	return nullptr;
}
// Regression
void* CreateDaalRidgeRegression(void* input) {
	try {
		auto in = static_cast<IInput<float> *>(input);
		return new(std::nothrow) RidgeRegression(*in);
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
		RidgeRegression* reg = static_cast<RidgeRegression *>(regression);
		auto in = static_cast<IInput<float> *>(input);
		reg->predict(*in);
	} CATCH_DAAL
}
const void* GetDaalRidgeRegressionBeta(void* regression) {
	try {
		RidgeRegression* reg = static_cast<RidgeRegression *>(regression);
		return &reg->getBeta();
	} CATCH_DAAL
	return nullptr;
}
const void* GetDaalRidgeRegressionPredictionData(void* regression) {
	try {
		RidgeRegression* reg = static_cast<RidgeRegression *>(regression);
		return &reg->getPredictionData();
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
		auto ridgeRegression = static_cast<RidgeRegression *>(regression);
		ridgeRegression->train();
	} CATCH_DAAL
}
void PredictDaalLinearRegression(void* regression, void* input) {
	try {
		LinearRegression* reg = static_cast<LinearRegression *>(regression);
		auto in = static_cast<IInput<float> *>(input);
		reg->predict(*in);
	} CATCH_DAAL
}
const void* GetDaalLinearRegressionBeta(void* regression) {
	try {
		LinearRegression* reg = static_cast<LinearRegression *>(regression);
		return &reg->getBeta();
	} CATCH_DAAL
	return nullptr;
}
const void* GetDaalLinearRegressionPredictionData(void* regression) {
	try {
		LinearRegression* reg = static_cast<LinearRegression *>(regression);
		return &reg->getPredictionData();
	} CATCH_DAAL
	return nullptr;
}


