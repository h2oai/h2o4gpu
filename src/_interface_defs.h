#ifndef INTERFACE_DEFS_H_
#define INTERFACE_DEFS_H_

#ifdef __MEX__

#define Printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);

#elif __R__

#define Printf Rprintf
extern "C" int Rprintf(const char* fmt, ...);

#else

#define Printf printf

#endif

#endif  // INTERFACE_DEFS_H_

