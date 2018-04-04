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
#include "./utils/defines.h"

namespace H2O4GPU {
namespace DAAL {

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

/* print out base class */
class PrintNumericTable {
protected:
	void print(const NumericTablePtr& input, const std::string& msg) const {
		printNumericTable(input, msg);
	}
};

/* IInput interface */
template<typename Input>
class IInput {
protected:
	NumericTablePtr _inputData;
	NumericTablePtr _featuresData;
	NumericTablePtr _dependentData;
public:
	virtual const SharedPtr<NumericTable>& getNumericTable() const {
		return this->_inputData;
	}
	virtual const NumericTablePtr& getFeaturesTable() const {
		return this->_featuresData;
	}
	virtual const NumericTablePtr& getDependentTable() const {
		return this->_dependentData;
	}

	virtual ~IInput() {}
};

/* HomogenousDaalData implementation for primitive data type: int, float, double */
template <typename Input=DEF_TYPE>
class HomogenousDaalData : public IInput<Input>
{
public:
	HomogenousDaalData(Input* inputArray, size_t m_dim, size_t n_dim) {
		if (!std::is_arithmetic<Input>::value)
			throw "Input data is not artithmetic type!";
		this->_inputData = SharedPtr<NumericTable>(new Matrix<Input>(m_dim, n_dim, inputArray));
		this->_featuresData = this->_inputData;
	}
	HomogenousDaalData(Input* features, size_t m_features, size_t n_features,
					Input* dependentVariables, size_t m_dependent, size_t n_dependent) {
		if (!std::is_arithmetic<Input>::value)
			throw "Input data is not artithmetic type!";
		this->_featuresData = NumericTablePtr(new Matrix<Input>(m_features, n_features));
		this->_dependentDAta = NuemricTablePtr(new Matrix<Input>(m_dependent, n_dependent));
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
		this->_featuresData = this->_inputData = dataSource.getNumericTable();
	}

	HomogenousDaalData(const std::string & filename, size_t features, size_t dependentVariables) {
		if (!this->fileExists(filename))
			throw "Input file doesn't exist!";
		if ("csv" != this->getFileExt(filename))
			throw "Input file isn't csv file format!";
		FileDataSource<CSVFeatureManager> dataSource(fileName,
						DataSource::doAllocateNumericTable,
						DataSource::doDictionaryFromContext);
		this->_featuresData =
				new HomogenNumericTable<>(features, 0, NumericTable::doNotAllocate);
		this->_dependentData =
				new HomogenNumericTable<>(dependentVariables, 0, NumericTable::doNotAllocate);
		NumericTablePtr mergedData(new MergedNumericTable(featuresData, dependentData));
		dataSource.loadDataBlock(mergedData.get());
		this->_featuresData = this->_inputData = dataSource.getNumericTable();
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
};

/* convert numericTable to raw pointer -> memory released in daal */
template<typename Output=double, size_t rows=0>
const Output* getRawOutput(const NumericTablePtr& nt) {
	rows = (rows == 0) ? (*nt.get()).getNumberOfRows() : rows;
	BlockDescriptor<Output> block;
	(*nt.get()).getBlockOfRows(0, rows, readOnly, block);
	const Output *array = block.getBlockPtr();
	(*nt.get()).releaseBlockOfRows(block);
	return array;
}


} /* end of DAAL namespace */
} /* end of H2O4GPU namespace */


#endif /* SRC_CPU_DAAL_INCLUDE_IINPUT_HPP_ */
