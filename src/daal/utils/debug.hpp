/*!
 * Copyright 2017 H2O.ai, Inc.
 * License Apache License Version 2.0 (see LICENSE for details)
 */
#ifndef SRC_CPU_DAAL_INCLUDE_DEBUG_HPP_
#define SRC_CPU_DAAL_INCLUDE_DEBUG_HPP_

#include <type_traits>
#include <typeinfo>
#include <memory>
#include <string>
#include <cstdlib>

#ifdef __GNUC__ /* if clang and GCC, use demangle for debug information  */
#	include <cxxabi.h>
#endif /* __GNUC__ */

template <typename T> std::string type_name() {
	typedef typename std::remove_reference<T>::type TR;
	std::unique_ptr<char, void(*)(void *) > self
			(
#ifdef __GNUC__
					abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
					nullptr,
#endif
					std::free
			);

	std::string rname = self != nullptr ? self.get() : typeid(TR).name();
	if (std::is_const<TR>::value) rname += " const";
	if (std::is_volatile<TR>::value) rname += " volatile";
	if (std::is_lvalue_reference<TR>::value) rname += "&";
	if (std::is_rvalue_reference<TR>::value) rname += "&&";
	return rname;
}


#endif /* SRC_CPU_DAAL_INCLUDE_DEBUG_HPP_ */
