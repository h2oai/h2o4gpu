/*
 * ridge_regression.h
 *
 *  Created on: Apr 7, 2018
 *      Author: monika
 */

#ifndef CPU_DAAL_RIDGE_REGRESSION_H_
#define CPU_DAAL_RIDGE_REGRESSION_H_

#include <daal.h>
#include <sys/stat.h>
#include <string>
#include "utils/defines.h"
#include "iinput.h"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::algorithms::ridge_regression;

class PUBLIC RidgeRegression {
public:
	typedef daal::algorithms::ridge_regression::training::ResultPtr TrainingResultPtr;

	RidgeRegression(IInput<double>* input);
	template<typename Input=FLOAT_TYPE>
		RidgeRegression(IInput<Input>& input);
	template<typename Input=FLOAT_TYPE>
		RidgeRegression(const IInput<Input>& input);
	const inline NumericTablePtr& getTrainingFeatures() const {
		return this->_featuresData;
	}
	const inline NumericTablePtr& getTrainingDependentData() const {
		return this->_dependentData;
	}
	void train();
	const NumericTablePtr& getBeta();
	const TrainingResultPtr& getModel() const;
	template<typename Input=FLOAT_TYPE>
	void predict(IInput<Input>& input);
	const NumericTablePtr& getPredictionData() const;

protected:
	TrainingResultPtr _trainingModel;
	NumericTablePtr _predictionData;
	NumericTablePtr _beta;
	NumericTablePtr _featuresData;
	NumericTablePtr _dependentData;
};

}} // H2O4GPU::DAAL namespace
#endif /* CPU_DAAL_RIDGE_REGRESSION_H_ */
