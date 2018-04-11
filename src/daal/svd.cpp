/*
 * svd.cpp
 *
 *  Created on: Apr 7, 2018
 *      Author: monika
 */
#include <utility>
#include "svd.h"

using namespace H2O4GPU::DAAL;

template<typename Input>
SVD::SVD(const IInput<Input>& input) {
	this->_origData = input.getNumericTable();
}
template<typename Input>
SVD::SVD(IInput<Input>& input) {
	this->_origData = std::move(input.getNumericTable());
}
void SVD::fit() {
	svd::Batch<> algorithm;
	algorithm.input.set(svd::data, this->_origData);
	algorithm.compute();
	this->_result = algorithm.getResult();
}

const typename SVD::resultPtr& SVD::getResult() const {
	return this->_result;
}
const NumericTablePtr& SVD::getSingularValues() {
	this->_singularValues = this->_result->get(svd::singularValues);
	return this->_singularValues;
}
const NumericTablePtr& SVD::getLeftSingularMatrix() {
	this->_leftSingularMatrix = this->_result->get(svd::leftSingularMatrix);
	return this->_leftSingularMatrix;
}
const NumericTablePtr& SVD::getRightSingularMatrix() {
	this->_rightSingularMatrix = this->_result->get(svd::rightSingularMatrix);
	return this->_rightSingularMatrix;
}

template SVD::SVD(const IInput<float>&);
template SVD::SVD(const IInput<int>&);
