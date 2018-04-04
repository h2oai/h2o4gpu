/*
 * defines.h
 *
 *  Created on: Mar 30, 2018
 *      Author: monika
 */

#ifndef SRC_CPU_DAAL_UTILS_DEFINES_H_
#define SRC_CPU_DAAL_UTILS_DEFINES_H_

#include <cstddef> // for size_t
#include <exception>
#include <cstdio>

#ifdef __cplusplus
#	define DLLEXPORT extern "C" __declspec(dllexport)
#else
#	define DLLEXPORT
#endif

#if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
#else
    #define DLL_PUBLIC
    #define DLL_LOCAL
#endif

namespace H2O4GPU {
namespace DAAL {
#define double DEF_TYPE;

template<typename Left, typename Right>
struct Either {
	union {
		Left left_value;
		Right right_value;
	};
	bool is_left;
};

} // end of DAAL namespace
} // end of H2O4GPU namespace


#endif /* SRC_CPU_DAAL_UTILS_DEFINES_H_ */
