/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef INTERFACE_DEFS_H_
#define INTERFACE_DEFS_H_

#ifdef __MEX__

#define Printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);

#elif __R__

#define Printf Rprintf
extern "C" void Rprintf(const char* fmt, ...);

#else

#define Printf printf

#endif

#endif  // INTERFACE_DEFS_H_

