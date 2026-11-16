// cuRAND host API names used by h2o4gpu, mapped to hipRAND.
#pragma once

#include "cuda_runtime.h"
#include <hiprand/hiprand.h>

#define curandGenerator_t hiprandGenerator_t
#define curandStatus_t hiprandStatus_t
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_MTGP32 HIPRAND_RNG_PSEUDO_MTGP32
#define CURAND_RNG_QUASI_SOBOL64 HIPRAND_RNG_QUASI_SOBOL64

#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerate hiprandGenerate
#define curandGenerateLongLong hiprandGenerateLongLong
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
