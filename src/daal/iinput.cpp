/*
 * iinput.cpp
 *
 *  Created on: Apr 4, 2018
 *      Author: monika
 */


#include "iinput.h"

#include <stddef.h>
#include <string>
#include <daal.h>
#include <type_traits>
#include <sys/stat.h>
#include <vector>
#include <memory>
#include "utils/service.h"
#include "utils/defines.h"
#include "utils/debug.hpp"

using namespace H2O4GPU::DAAL;
using namespace daal::services;
using namespace daal::data_management;

/* Base print class */
void PrintTable::print(const NumericTablePtr& input, const std::string& msg, size_t rows, size_t cols) const {
	printNumericTable(input, msg.c_str(), rows, cols);
}

void PrintTable::print(NumericTable& input, const std::string& msg, size_t rows, size_t cols) {
	printNumericTable(input, msg.c_str(), rows, cols);
}

/* generic IInput interface */
template<typename Input>
const typename IInput<Input>::NTPtr& IInput<Input>::getNumericTable() const {
	return this->_inputData;
}
const NumericTablePtr& IInput<float>::getNumericTable() const {
	return this->_inputData;
}
template<typename Input>
const typename IInput<Input>::NTPtr& IInput<Input>::getFeaturesTable() const {
	return this->_featuresData;
}
const NumericTablePtr& IInput<float>::getFeaturesTable() const {
	return this->_featuresData;
}
template<typename Input>
const typename IInput<Input>::NTPtr& IInput<Input>::getDependentTable() const {
	return this->_dependentData;
}
const NumericTablePtr& IInput<float>::getDependentTable() const {
	return this->_dependentData;
}
/* HomogenousDaalData specialized */
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(Input* inputArray, size_t m_dim, size_t n_dim) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_inputData =
			NTPointer(new Matrix<Input>(m_dim, n_dim, inputArray));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(const Input* inputArray, size_t m_dim, size_t n_dim) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	auto input = const_cast<Input*>(inputArray);
	this->_inputData =
			NTPointer(new Matrix<Input>(m_dim, n_dim, input));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(Input* features, size_t m_features, size_t n_features,
		Input* dependentData, size_t m_data, size_t n_data) {
	if (!std::is_arithmetic<Input>::value)
			throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_featuresData =
			NTPointer(new Matrix<Input>(m_features, n_features, features));
	this->_dependentData =
			NTPointer(new Matrix<Input>(m_data, n_data, dependentData));
	//this->_inputData = NTPointer(new MergedNumericTable(this->_featuresData, this->_dependentData));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(const Input* features, size_t m_features, size_t n_features,
		const Input* dependentData, size_t m_data, size_t n_data) {
	if (!std::is_arithmetic<Input>::value)
			throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_featuresData =
			NTPointer(new Matrix<Input>(m_features, n_features, const_cast<Input*>(features)));
	this->_dependentData =
			NTPointer(new Matrix<Input>(m_data, n_data, const_cast<Input*>(dependentData)));
	//this->_inputData = NTPointer(new MergedNumericTable(this->_featuresData, this->_dependentData));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(std::vector<Input>& input, size_t m_dim, size_t n_dim) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	auto inputArray = input.data();
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_inputData =
			NTPointer(new Matrix<Input>(m_dim, n_dim, inputArray));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(std::vector<Input>& features, size_t m_features, size_t n_features,
		std::vector<Input>& dependentData, size_t m_data, size_t n_data) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_featuresData =
			NTPointer(new Matrix<Input>(m_features, n_features, features.data()));
	this->_dependentData =
			NTPointer(new Matrix<Input>(m_data, n_data, dependentData.data()));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(const std::vector<Input>& input, size_t m_dim, size_t n_dim) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	auto inputArray = const_cast<std::vector<Input> *>(&input)->data();
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	this->_featuresData =
			NTPointer(new Matrix<Input>(m_dim, n_dim, inputArray));
}
template<typename Input>
HomogenousDaalData<Input>::HomogenousDaalData(const std::vector<Input>& features, size_t m_features, size_t n_features,
		const std::vector<Input>& dependentData, size_t m_data, size_t n_data) {
	if (!std::is_arithmetic<Input>::value)
		throw "Input type is not arithmetic!";
	using NTPointer = typename HomogenousDaalData<Input>::NTPtr;
	auto featuresArray = const_cast<std::vector<Input> *>(&features)->data();
	auto dependentDataArray = const_cast<std::vector<Input> *>(&dependentData)->data();
	this->_featuresData =
			NTPointer(new Matrix<Input>(m_features, n_features, featuresArray));
	this->_dependentData =
			NTPointer(new Matrix<Input>(m_data, n_data, dependentDataArray));
}
/* IInput interface accepting CSV files or arrays */
HomogenousDaalData<std::string>::HomogenousDaalData(const std::string& filename) {
	if (!this->fileExists(filename))
		throw "Input file doesn't exist!";
	if ("csv" != this->getFileExt(filename))
		throw "Input file isn't in csv file format!";
	FileDataSource<CSVFeatureManager> dataSource(filename,
							DataSource::doAllocateNumericTable,
							DataSource::doDictionaryFromContext);
	// retrieve data from the input file
	dataSource.loadDataBlock();
	this->_inputData = dataSource.getNumericTable();
}
HomogenousDaalData<std::string>::HomogenousDaalData(const std::string & filename, size_t features, size_t dependentVariables)
{
	if (!this->fileExists(filename))
			throw "Input file doesn't exist!";
		if ("csv" != this->getFileExt(filename))
			throw "Input file isn't in csv file format!";
		FileDataSource<CSVFeatureManager> dataSource(filename,
								DataSource::doAllocateNumericTable,
								DataSource::doDictionaryFromContext);

		this->_featuresData = NumericTablePtr(
				new HomogenNumericTable<>(features, 0, NumericTable::doNotAllocate));
		this->_dependentData = NumericTablePtr(
				new HomogenNumericTable<>(dependentVariables, 0, NumericTable::doNotAllocate));
		this->_inputData = NumericTablePtr(
				new MergedNumericTable(this->_featuresData, this->_dependentData));

		dataSource.loadDataBlock(this->_inputData.get());
}


template class IInput<float>;
template class IInput<int>;
template class HomogenousDaalData<float>;
template class HomogenousDaalData<int>;
template class HomogenousDaalData<std::string>;
