/*!
 * Copyright 2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(code) __CUDA_CHECK(code, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define __CUDA_CHECK(code, file, line, func) do \
{ \
    cudaError_t err = code; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s [%s:%d in %s]\n", cudaGetErrorString(err),  file, line, func); \
        exit(33); \
    } \
} while(0)