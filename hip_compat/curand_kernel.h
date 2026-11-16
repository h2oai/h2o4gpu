// cuRAND device API names used by h2o4gpu, mapped to hipRAND.
#pragma once

#include "curand.h"
#include <hiprand/hiprand_kernel.h>

#define curandState hiprandState
#define curand_init hiprand_init
#define curand_uniform hiprand_uniform
#define curand_uniform_double hiprand_uniform_double
