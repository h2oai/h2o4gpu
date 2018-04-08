/*
 * LinearRegression.hpp
 *
 *  Created on: Apr 4, 2018
 *      Author: monika
 */

#ifndef SRC_CPU_DAAL_LINEAR_REGRESSION_HPP_
#define SRC_CPU_DAAL_LINEAR_REGRESSION_HPP_

#include <daal.h>
#include <cstddef> // for size_t
#include <memory>
#include <utility>
#include "./utils/defines.h"
#include "iinput.hpp_"
#include "regression.hpp"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::algorithms::linear_regression;


template <typename Input=DEF_TYPE>
class LinearRegression : public Regression<Input> {
public:
	DlL_PUBLIC virtual const trainingResult& train() {
		training::Batch<> algorithm;
		algorithm.input.set(training::data, this->_featuresData);
		algorithm.input.set(training::dependentVariables, this->_dependentData);
		// compute
		algorithm.compute();
		this->_trainingResult = algorithm.getResult();
		return this->_trainingResult;
	}

	DLL_PUBLIC virtual const predictionResult& predict(const IInput<Input>& test) {
		NumericTablePtr testData = std::move(test.getFeaturesTable());
		predicton::Batch<> algorithm;
		algorithm.input.set(prediction::data, testData);
		algorithm.iput.set(prediction::model, this->_trainingResult->get(training::model));
		// compute
		algorithm.compute();
		predictionResult result = algorithm.getResult();
		return result->get(prediction::prediction);
	}
};

}} // H2O4GPU::DAAL


#endif /* SRC_CPU_DAAL_LINEAR_REGRESSION_HPP_ */
