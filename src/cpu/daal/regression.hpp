/*
 * regression.hpp
 *
 *  Created on: Apr 4, 2018
 *      Author: monika
 */

#ifndef SRC_CPU_DAAL_REGRESSION_HPP_
#define SRC_CPU_DAAL_REGRESSION_HPP_

#include <daal.h>
#include <cstddef> // for size_t
#include <memory>
#include <utility>
#include "iinput.hpp"
#include "./utils/defines.h"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::algorithms::regression;

template<typename Input=DEF_TYPE>
class Regression : public PrintNumericTable {
public:
	enum Type { linear, ridge };
	typedef training::ResultPtr trainingResult;
	typedef prediction::ResultPtr predictionResult;
	virtual ~Regresion() {}
	DLL_PUBLIC Regression() {}
	DLL_PUBLIC Regression(const IInput<Input>& input) {
		this->_featuresData = input.getFeaturesTable();
		this->_dependentData = input.getDependentTable();
	}
	DLL_PUBLIC virtual void SetInput(IInput<Input>& input) {
		this->_featuresData = std::move(input.getFeaturesTable());
		this->_dependentData = std::move(input.getDependentTable());
	}
	DLL_PUBLIC virtual const trainingResult& train() = 0;
	DLL_PUBLIC virtual const predictionResult& predict(const IInput<Input>& input) = 0;
protected:
	NumericTablePtr _featuresData;
	NumericTablePtr _dependentData;
	NumericTablePtr _beta;
	trainingResult _trainingResult;
	predictionResult _predictionResult;
};



#endif /* SRC_CPU_DAAL_REGRESSION_HPP_ */
