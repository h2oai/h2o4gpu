#include "device_utilities.h"

__global__ void fp32Array2fp16Array(const float * fp32Array, half* fp16Array,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		fp16Array[i] =  __float2half(fp32Array[i]);
	}
}

__global__ void fp16Array2fp32Array(float * fp32Array, const half* fp16Array,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		fp32Array[i] =  __half2float(fp16Array[i]);
	}
}