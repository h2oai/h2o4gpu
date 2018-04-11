/*
 * svd.h
 *
 *  Created on: Apr 7, 2018
 *      Author: monika
 */

#ifndef CPU_DAAL_SVD_H_
#define CPU_DAAL_SVD_H_

#include <daal.h>
#include <stddef.h>
#include "utils/defines.h"
#include "iinput.h"

namespace H2O4GPU {
namespace DAAL {

typedef float FLOAT_TYPE;

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

class PUBLIC SVD {
public:
	typedef svd::ResultPtr resultPtr;

	template<typename Input=FLOAT_TYPE>
	SVD(const IInput<Input>& input);
	template<typename Input=FLOAT_TYPE>
	SVD(IInput<Input>& input);
	void fit();
	const resultPtr& getResult() const;
	const NumericTablePtr& getSingularValues();
	const NumericTablePtr& getRightSingularMatrix();
	const NumericTablePtr& getLeftSingularMatrix();

protected:
	NumericTablePtr _origData;
	resultPtr _result;
	NumericTablePtr _singularValues;
	NumericTablePtr _rightSingularMatrix;
	NumericTablePtr _leftSingularMatrix;
};

}} // H2O4GPU::DAAL namespace



#endif /* CPU_DAAL_SVD_H_ */
