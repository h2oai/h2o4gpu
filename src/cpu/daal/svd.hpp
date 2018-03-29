/*!
 * Copyright 2017 H2O.ai, Inc.
 * License Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef SRC_CPU_DAAL_INCLUDE_SVD_H_
#define SRC_CPU_DAAL_INCLUDE_SVD_H_

#include <daal.h>
#include "iinput.hpp"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::algorithms;

class SVD {
public:
	typedef svd::ResultPtr resultPtr;

	SharedPtr<NumericTable> _origData;
	resultPtr _result;
	NumericTablePtr _singularValues;
	NumericTablePtr _rightSingularMatrix;
	NumericTablePtr _leftSingularMatrix;

	template<typename Input>
	SVD(const IInput<Input>& input) {
		this->_origData;
	}

	void fit() {
		svd::Batch<> algorithm;
		algorithm.input.set(svd::data, this->_origData);

		// Compute SVD
		algorithm.compute();
		this->_result = algorithm.getResult();
		this->_result->get(svd::singularValues);
	}

	const NumericTablePtr getSingularValues() const
	{
		return this->_result->get(svd::singularValues);
	}

	const NumericTablePtr getRightSingularMatrix() const
	{
		return this->_result->get(svd::rightSingularMatrix);
	}

	const NumericTablePtr getLeftSingularMatrix() const
	{
		return this->_result->get(svd::leftSingularMatrix);
	}
};

} /* end of namespace DAAL
} /* end of namespace H2O4GPU

#endif /* SRC_CPU_DAAL_INCLUDE_SVD_H_ */
