/*
 * ridge_regression.cpp
 *
 *  Created on: Apr 7, 2018
 *      Author: monika
 */

#include <utility>
#include "ridge_regression.h"

using namespace H2O4GPU::DAAL;

RidgeRegression::RidgeRegression(IInput<double>* input) {
	this->_featuresData = std::move(input->getFeaturesTable());
	this->_dependentData = std::move(input->getDependentTable());
}
template<typename Input>
RidgeRegression::RidgeRegression(IInput<Input>& input) {
	this->_featuresData = std::move(input.getFeaturesTable());
	this->_dependentData = std::move(input.getDependentTable());
}
template<typename Input>
RidgeRegression::RidgeRegression(const IInput<Input>& input) {
	this->_featuresData = input.getFeaturesTable();
	this->_dependentData = input.getDependentTable();
}
void RidgeRegression::train() {
	training::Batch<> algorithm;
	algorithm.input.set(training::data, this->_featuresData);
	algorithm.input.set(training::dependentVariables, this->_dependentData);
	algorithm.compute();
	this->_trainingModel = algorithm.getResult();
}
const NumericTablePtr& RidgeRegression::getBeta() {
	this->_beta = this->_trainingModel->get(training::model)->getBeta();
	return this->_beta;
}
const typename RidgeRegression::TrainingResultPtr& RidgeRegression::getModel() const {
	return this->_trainingModel;
}
template<typename Input>
void RidgeRegression::predict(IInput<Input>& input) {
	auto testData = std::move(input.getFeaturesTable());
	prediction::Batch<> algorithm;
	algorithm.input.set(prediction::data, testData);
	algorithm.input.set(prediction::model, this->_trainingModel->get(training::model));
	algorithm.compute();
	auto predictionResult = algorithm.getResult();
	this->_predictionData = predictionResult->get(prediction::prediction);
}
const NumericTablePtr& RidgeRegression::getPredictionData() const {
	return this->_predictionData;
}

template RidgeRegression::RidgeRegression<double>(IInput<double> &);
template RidgeRegression::RidgeRegression<double>(const IInput<double> &);
template RidgeRegression::RidgeRegression<float>(IInput<float> &);
template RidgeRegression::RidgeRegression<float>(const IInput<float> &);
template void RidgeRegression::predict<double>(IInput<double> &);
template void RidgeRegression::predict<float>(IInput<float> &);
