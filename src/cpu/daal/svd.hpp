/*!
 * Copyright 2017 H2O.ai, Inc.
 * License Apache License Version 2.0 (see LICENSE for details)
 */

#ifndef SRC_CPU_DAAL_INCLUDE_SVD_H_
#define SRC_CPU_DAAL_INCLUDE_SVD_H_

#include <daal.h>
#include <cstddef> // for size_t
#include <memory>
#include "iinput.hpp"
#include "./utils/defines.h"

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

	template<typename Input=DEF_TYPE>
	DLL_PUBLIC SVD(const IInput<Input>& input) {
		this->_origData = input.getNumericTable();
	}

	template<typename Input=DEF_TYPE>
	DLL_PUBLIC SVD(Input* inputArray, size_t m_dim, size_t n_dim) {
		auto table_nt = std::make_shared<HomogenousDaalData<Input> >(inputArray, m_dim, n_dim);
		this->_origData = table_nt->getNumericTable();
	}

	DLL_PUBLIC void fit() {
		svd::Batch<> algorithm;
		algorithm.input.set(svd::data, this->_origData);

		// Compute SVD
		algorithm.compute();
		this->_result = algorithm.getResult();
	}

	DLL_PUBLIC const NumericTablePtr& getSingularValues() const
	{
		return this->_result->get(svd::singularValues);
	}

	DLL_PUBLIC const NumericTablePtr& getRightSingularMatrix() const
	{
		return this->_result->get(svd::rightSingularMatrix);
	}

	DLL_PUBLIC const NumericTablePtr& getLeftSingularMatrix() const
	{
		return this->_result->get(svd::leftSingularMatrix);
	}
};

} /* end of namespace DAAL
} /* end of namespace H2O4GPU

#endif /* SRC_CPU_DAAL_INCLUDE_SVD_H_ */
