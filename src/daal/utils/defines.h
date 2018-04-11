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

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define PUBLIC __attribute__ ((dllexport))
    #else
      #define PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define PUBLIC __attribute__ ((dllimport))
    #else
      #define PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define LOCAL
#else
  #if __GNUC__ >= 4
    #define PUBLIC __attribute__ ((visibility ("default")))
    #define LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define PUBLIC
    #define LOCAL
  #endif
#endif

namespace H2O4GPU {
namespace DAAL {

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
