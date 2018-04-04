/*
 * h2o4gpu_daal_c.h
 *
 *  Created on: Apr 2, 2018
 *      Author: monika
 */

#ifndef SRC_CPU_DAAL_H2O4GPU_DAAL_C_H_
#define SRC_CPU_DAAL_H2O4GPU_DAAL_C_H_

#include <daal.h>
#include <cstddef> // for size_t
#include <string>
#include <iostream>
#include <exception>
#include <stdexcept>
#include "utils/defines.h"
#include "iinput.hpp"
#include "svd.hpp"
#include <sstream>


using namespace H2O4GPU::DAAL;

#define CATCH_DAAL 																				\
	catch(const std::runtime_error &e) { 														\
		fprintf(stderr, "Runtime error: %s in %s at line %d", e.what(), __FILE__, __LINE__); 	\
	} catch(const std::exception &e) {															\
		fprintf(stderr, "Error occurred: %s in %s at line %d", e.what(), __FILE__, __LINE__); 	\
	} catch (...) {																				\
		fprintf(stderr, "Unknown failure occurred. Possible memory corruption.");				\
	}

//input to enter algorithms
void *CreateDaalInput(double*, size_t, size_t);
void *CreateDaalInput(double*, size_t, size_t, double*, size_t, size_t);
void *CreateDaalInput(const std::string&);
void *CreateDaalInput(const std::string&, size_t size_t)
void *CreateDaalInput(const std::string&, size_t, size_t, size_t, size_t);

template<typename T=DEF_TYPE>
void DeleteDaalInput(void* input) {
	delete static_cast<IInput<T> *>(input);
}
template<typename T=DEF_TYPE>
void printDaalInput(void* input, const char* msg) {
	try {
		auto di = static_cast<const HomogenousDaalData<T> *>(input);
		auto pnt = di->getNumericTable();
		printNumericTable(pnt, msg);
	} CATCH_DAAL
}
template<typename T=DEF_TYPE>
void* getDaalNT(const void* daalInput) {
	try {
		auto di = static_cast<const HomogenousDaalData<T> *>(daalInput);
		auto pnt = di->getNumericTable();
		return pnt;
	} CATCH_DAAL
	return nullptr;
}
void printDaalNT(const void*,const char*, size_t, size_t);


// Singular value decomposition algorithm
void* CreateDaalSVD(const void*);
void* CreateDaalSVD(double*, size_t, size_t);
void DeleteDaalSVD(void*);
void fitDaalSVD(const void*);
const void* getDaalSigmas(const void*) const;
const void* getDaalRightSingularMatrix(const void*) const;
const void* getDaalLeftSingularMatrix(const void*) const;

// Regression
void* CreateDaalRegression(const void*);
void* CreateDaalRegression();
void DeleteDaalRegression();
const void* train(Regression::Type);
const void* predict(Regression::Type, const void*);
void setInput(const void*);






#endif /* SRC_CPU_DAAL_H2O4GPU_DAAL_C_H_ */
