#ifndef INTERFACE_DEFS_H_
#define INTERFACE_DEFS_H_

#ifdef __MEX__
#define printf mexPrintf
extern "C" int mexPrintf(const char* fmt, ...);
#endif  // __MEX__

#ifdef __R__
#define printf Rprintf
extern "C" int Rprintf(const char* fmt, ...);
#endif  // __R__

#endif  // INTERFACE_DEFS_H_

