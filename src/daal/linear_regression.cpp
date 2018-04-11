/*
 * linear_regression.cpp
 *
 *  Created on: Apr 7, 2018
 *      Author: monika
 */


#include <utility>
#include "linear_regression.h"

using namespace H2O4GPU::DAAL;
using namespace daal;
using namespace daal::algorithms::linear_regression;

template<typename Input>
LinearRegression::LinearRegression(IInput<Input>& input) {
	this->_featuresData = std::move(input.getFeaturesTable());
	this->_dependentData = std::move(input.getDependentTable());
}
template<typename Input>
LinearRegression::LinearRegression(const IInput<Input>& input) {
	this->_featuresData = input.getFeaturesTable();
	this->_dependentData = input.getDependentTable();
}
void LinearRegression::train() {
	training::Batch<> algorithm;
	algorithm.input.set(training::data, this->_featuresData);
	algorithm.input.set(training::dependentVariables, this->_dependentData);
	algorithm.compute();
	this->_trainingModel = algorithm.getResult();
}
const typename LinearRegression::TrainingResultPtr& LinearRegression::getModel() const {
	return this->_trainingModel;
}
template<typename Input>
void LinearRegression::predict(IInput<Input>& input) {
	auto testData = std::move(input.getFeaturesTable());
	prediction::Batch<> algorithm;
	algorithm.input.set(daal::algorithms::linear_regression::prediction::data, testData);
	algorithm.input.set(daal::algorithms::linear_regression::prediction::model, this->_trainingModel->get(daal::algorithms::linear_regression::training::model));
	algorithm.compute();
	auto predictionResult = algorithm.getResult();
	this->_predictionData = predictionResult->get(prediction::prediction);
}
const NumericTablePtr& LinearRegression::getPredictionData() const {
	return this->_predictionData;
}
const NumericTablePtr& LinearRegression::getBeta() {
	this->_beta = this->_trainingModel->get(training::model)->getBeta();
	return this->_beta;
}
template LinearRegression::LinearRegression(IInput<float> &);
template LinearRegression::LinearRegression(IInput<int> &);
template LinearRegression::LinearRegression(const IInput<float> &);
template LinearRegression::LinearRegression(const IInput<int> &);
template void LinearRegression::predict(IInput<float>&);
template void LinearRegression::predict(IInput<int>&);


