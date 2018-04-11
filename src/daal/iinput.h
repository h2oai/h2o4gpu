/*
 * iinput.h
 *
 *  Created on: Apr 4, 2018
 *      Author: monika
 */

#ifndef CPU_DAAL_IINPUT_H_
#define CPU_DAAL_IINPUT_H_

#include <string>
#include <daal.h>
#include <stddef.h>
#include <vector>
#include <sys/stat.h>
//#include "utils/service.h"
#include "utils/defines.h"

namespace H2O4GPU {
namespace DAAL {

typedef float FLOAT_TYPE;

using namespace daal;
using namespace daal::data_management;
using namespace daal::services;

/* Base class to print numeric table */
class PUBLIC PrintTable {
public:
	virtual void print(const NumericTablePtr& input, const std::string& ="", size_t =0, size_t =0) const;
	virtual void print(NumericTable& input, const std::string& ="", size_t =0, size_t =0);
	virtual ~PrintTable() {}
};

/* array (numpy) input */
template<typename Input>
class PUBLIC IInput : public PrintTable {
public:
	typedef SharedPtr<HomogenNumericTable<Input> > NTPtr;
protected:
	NTPtr _inputData;
	NTPtr _featuresData;
	NTPtr _dependentData;
public:
	const NTPtr& getNumericTable() const;
	const NTPtr& getFeaturesTable() const;
	const NTPtr& getDependentTable() const;
};

template<>
class PUBLIC IInput<float> : public PrintTable {
public:
	typedef NumericTablePtr NTPtr;
protected:
	NTPtr _inputData;
	NTPtr _featuresData;
	NTPtr _dependentData;
public:
	const NTPtr& getNumericTable() const;
	const NTPtr& getFeaturesTable() const;
	const NTPtr& getDependentTable() const;
};

template<typename Input=FLOAT_TYPE>
class PUBLIC HomogenousDaalData : public IInput<Input>
{
public:
	typedef Input Type;
	HomogenousDaalData(Input* inputArray, size_t m_dim, size_t n_dim);
	HomogenousDaalData(const Input* inputArray, size_t m_dim, size_t n_dim);
	HomogenousDaalData(Input* features, size_t m_features, size_t n_features,
			Input* dependentData, size_t m_data, size_t n_data);
	HomogenousDaalData(const Input* features, size_t m_features, size_t n_features,
			const Input* dependentData, size_t m_data, size_t n_data);
	HomogenousDaalData(std::vector<Input>& input, size_t m_dim, size_t n_dim);
	HomogenousDaalData(std::vector<Input>&, size_t, size_t, std::vector<Input>&, size_t, size_t);
	HomogenousDaalData(const std::vector<Input>&, size_t, size_t);
	HomogenousDaalData(const std::vector<Input>&, size_t, size_t, const std::vector<Input>&, size_t, size_t);
};

/* IInput interface accepting CSV files or arrays */
template<>
class PUBLIC HomogenousDaalData<std::string> : public IInput<FLOAT_TYPE>
{
public:
	typedef std::string Type;
	HomogenousDaalData(const std::string& filename);
	HomogenousDaalData(const std::string & filename, size_t features, size_t dependentVariables);
	inline bool fileExists(const std::string & filename) {
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
}} // H2O4GPU::DAAL


#endif /* CPU_DAAL_IINPUT_H_ */
