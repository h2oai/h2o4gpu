#pragma once

#include "../../../cuda_runtime.h"
#include <thrust/system/hip/error.h>

THRUST_NAMESPACE_BEGIN
inline const error_category &cuda_category() { return hip_category(); }
THRUST_NAMESPACE_END
