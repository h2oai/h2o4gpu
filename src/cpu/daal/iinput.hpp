/*!
 * Copyright 2017 H2O.ai, Inc.
 * License Apache License Version 2.0 (see LICENSE for details)
 * purpose: Interface for entering input to DAAL library
 * 			from raw pointers and csv file to NumericTable (needed for DAAL) calculation
 *			copy ctor and eq. operator based on default behavior (shared_ptr points to the same object)
 */

#ifndef SRC_CPU_DAAL_INCLUDE_IINPUT_HPP_
#define SRC_CPU_DAAL_INCLUDE_IINPUT_HPP_

#include <daal.h>
#include <type_traits> // for primitive data types
#include <iostream>
#include <sys/stat.h>
#include <string>

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

/* IInput interface */
template<typename Input>
class IInput {
protected:
	SharedPtr<NumericTable> _inputData;
public:
	virtual const SharedPtr<NumericTable>& getNumericTable() const {
		return this->_inputData;
	}
	virtual ~IInput() {}
};

/* HomogenousDaalData implementation for primitive data type: int, float, double */
template <typename Input>
class HomogenousDaalData : public IInput<Input>
{
public:
	HomogenousDaalData(Input* const inputArray, size_t m_dim, size_t n_dim) {
		if (!std::is_arithmetic<Input>::value)
			throw "Input data is not artithmetic type!";
		this->_inputData = SharedPtr<NumericTable>(new Matrix<Input>(m_dim, n_dim, inputArray));
	}
	virtual ~HomogenousDaalData() {}
};

/* HomogenousDaalData implementation for csv file input */
template <>
class HomogenousDaalData<std::string> : public IInput<std::string>
{
public:
	HomogenousDaalData(const std::string &filename) {
		if (!this->fileExists(filename))
			throw "Input file doesn't exist!";
		if ("csv" != this->getFileExt(filename))
			throw "Input file isn't csv file format!";
		FileDataSource<CSVFeatureManager> dataSource(fileName,
						DataSource::doAllocateNumericTable,
						DataSource::doDictionaryFromContext);
		// retrieve data from the input file
		dataSource.loadDataBlock();
		this->_inputData = dataSource.getNumericTable();
	}

	inline bool fileExists(const std::string &filename) {
		struct stat buffer;
			return (stat (filename.c_str(), &buffer) == 0);
	}

	inline std::string getFileExt(const std::string &filename) {
		size_t i = filename.rfind('.', filename.length());
			if (i != std::string::npos) {
				return(filename.substr(i+1, filename.length() - i));
			}
		return("");
	}
}

/* convert numericTable to raw pointer -> memory released in daal */
template<typename Output, size_t rows, size_t columns>
const Output* getRawOutput(const NumericTablePtr& nt) {
	BlockDescriptor<Output> block;
	(*nt.get()).getBlockOfRows(0, rows, readOnly, block);
	const Output *array = block.getBlockPtr();
	(*nt.get()).releaseBlockOfRows(block);
	return array;
}

template<typename Output>
const Output* getRawOutput(const NumericTablePtr& nt) {
	BlockDescriptor<double> block;
	auto rows = (*nt.get()).getNumberOfRows();
	(*nt.get()).getBlockOfRows(0, rows, readOnly, block);
	const double *array = block.getBlockPtr();
	(*nt.get()).releaseBlockOfRows(block);
	return array;
}

} /* end of DAAL namespace */
} /* end of H2O4GPU namespace */

#endif /* SRC_CPU_DAAL_INCLUDE_IINPUT_HPP_ */
