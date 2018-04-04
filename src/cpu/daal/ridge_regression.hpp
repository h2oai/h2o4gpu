/*
 * regression.hpp
 *
 *  Created on: Apr 4, 2018
 *      Author: monika
 */

#ifndef SRC_CPU_DAAL_RIDGE_REGRESSION_HPP_
#define SRC_CPU_DAAL_RIDGE_REGRESSION_HPP_

#include <daal.h>
#include <cstddef> // for size_t
#include <memory>
#include <utility>
#include "iinput.hpp"
#include "./utils/defines.h"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::algorithms::ridge_regression;

template <typename Input=DEF_TYPE>
class RidgeRegression : public Regression<Input> {
public:
	DLL_PUBLIC virtual const trainingResult& train() {
		training::Batch algorithm;
		algorithm.input.set(training::data, this->_featuresData);
		algorithm.input.set(training::dependentVariables, this->_dependentData);
		// compute
		algorithm.compute();
		this->_trainingResult = algorithm.getResult();
		return this->_trainingResult;
	}

	DLL_PUBLIC const NumericTablePtr& getBeta() const {
		this->_trainingResult->get(training::model)->getBeta();
		return this->_trainingResult;
	}
	DLL_PUBLIC virtual const predictionResult& predict(IInput<Input>& input) {
		NumericTablePtr testData = std::move(input.getFeaturesTable());
		prediction::Batch<> algorithm;
		algorithm.input.set(prediction::data, testData);
		algorithm.input.set(prediction::model, this->_trainingResult->get(training::model));
		// compute
		algorithm.compute();
		predictionResult result = algorithm.getResult();
		return result->get(prediction::prediction);
	}
};



#endif /* SRC_CPU_DAAL_RIDGE_REGRESSION_HPP_ */
