/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef UTILITIES_CUH
#define UTILITIES_CUH

extern "C" int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);
extern "C" void cusolveSafeCall(cusolverStatus_t);

#endif
