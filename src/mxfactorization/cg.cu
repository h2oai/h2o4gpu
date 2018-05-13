/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * cg.cu
 *  Created on: July 22, 2016
 *  Author: Wei Tan (wtan@us.ibm.com)
 *  CUDA kernels related to batch CG solver used in ALS
 *	CG solver: https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */

#include "als.h"
#include "device_utilities.h"
#include "host_utilities.h"
#include <fstream>
#define SCAN_BATCH 24
#define CG_ERROR 1e-4
#undef DEBUG

//CG (iterative solve) kernel
//each block solves a A*x=b 
__global__ void updateXWithCGKernel(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];

	//sharedx<--x
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	//sharedx[threadIdx.x] = 0;
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int i = 0; i < f; i++)
		//this is math correct and coalesced because A is symmetric
		temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedx[i];
	sharedr[threadIdx.x] = b[blockIdx.x*blockDim.x + threadIdx.x] - temp;
	//p=r;
	sharedp[threadIdx.x] = sharedr[threadIdx.x];
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = sharedr[threadIdx.x]
			*sharedr[threadIdx.x];
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		temp = 0;
		for(int i = 0; i < f; i++)
			//this is math correct and coalesced because A is symmetric
			temp += A[blockIdx.x*f*f + f*i + threadIdx.x]*sharedp[i];
		sharedap[threadIdx.x] = temp;
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = sharedp[threadIdx.x]
				*sharedap[threadIdx.x];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		//x=x+alpha*p;
		sharedx[threadIdx.x] = 
			sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        //r=r-alpha*Ap;
		sharedr[threadIdx.x] = 
			sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		//p=r+(rsnew/rsold)*p;
		sharedp[threadIdx.x] = 
			sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	//x<--sharedx
	x[blockIdx.x*blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

//CG (iterative solve) kernel
//each block solves a A*x=b and A in fp16
__global__ void updateXWithCGKernel3(half * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];

	//sharedx<--x
	sharedx[threadIdx.x] = x[blockIdx.x*blockDim.x + threadIdx.x];
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int i = 0; i < f; i++)
		//this is math correct and coalesced because A is symmetric
		temp += __half2float(A[blockIdx.x*f*f + f*i + threadIdx.x])*sharedx[i];
	sharedr[threadIdx.x] = b[blockIdx.x*blockDim.x + threadIdx.x] - temp;
	//p=r;
	sharedp[threadIdx.x] = sharedr[threadIdx.x];
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = sharedr[threadIdx.x]
			*sharedr[threadIdx.x];
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		temp = 0;
		for(int i = 0; i < f; i++)
			//this is math correct and coalesced because A is symmetric
			temp += __half2float(A[blockIdx.x*f*f + f*i + threadIdx.x])*sharedp[i];
		sharedap[threadIdx.x] = temp;
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = sharedp[threadIdx.x]
				*sharedap[threadIdx.x];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		//x=x+alpha*p;
		sharedx[threadIdx.x] = 
			sharedx[threadIdx.x] + alpha[0] * sharedp[threadIdx.x];
        //r=r-alpha*Ap;
		sharedr[threadIdx.x] = 
			sharedr[threadIdx.x] - alpha[0] * sharedap[threadIdx.x];
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		//p=r+(rsnew/rsold)*p;
		sharedp[threadIdx.x] = 
			sharedr[threadIdx.x] + beta[0] * sharedp[threadIdx.x];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	//x<--sharedx
	x[blockIdx.x*blockDim.x + threadIdx.x] = sharedx[threadIdx.x];
}

//blockDim.x=64 or 96 (two or three WARPs) instead of 100 -- WARP shuffle seems requiring this
__global__ void updateXWithCGKernel2(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	extern __shared__ float smem[];
	float *sharedx = &smem[0];
	float *sharedp = &smem[f];
	float *sharedr = &smem[2*f];
	float *sharedap = &smem[3*f];
	float *rsold = &smem[4*f]; 
	float *alpha = &smem[4*f+1];
	float *rsnew = &smem[4*f+2];
	float *beta = &smem[4*f+3];

	//sharedx<--x
	for(int k = threadIdx.x; k < f; k += blockDim.x)
		sharedx[k] = x[blockIdx.x*f + k];
	__syncthreads();
	//r=b-A*x;
	float temp = 0;
	for(int k = threadIdx.x; k < f; k += blockDim.x){
		temp = 0;
		for(int i = 0; i < f; i++)
			temp += A[blockIdx.x*f*f + f*i + k]*sharedx[i];
		sharedr[k] = b[blockIdx.x*f + k] - temp;
		//p=r;
		sharedp[k] = sharedr[k];
	}
	//rsold=r'*r;
	if(threadIdx.x == 0){
		rsold[0] = 0;
	}
	temp = 0;
	for(int k = threadIdx.x; k < f; k += blockDim.x){
		temp += sharedr[k]*sharedr[k];
	}
	blockReduceSumWithAtomics(rsold, temp);	
    //temp = blockReduceSum(shared, temp);
	__syncthreads();
	#ifdef DEBUG
	if(threadIdx.x==0){
		printf("***rsold:\n");
		printf("rsold = %f \n", rsold[0]);
		printf("***shared memory content after 1st blockReduceSum:\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedp[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedr[i]);
		printf("\n");
		for(int i = 0; i < f; i++)
			printf("%f ", sharedap[i]);
		printf("\n");
	}
	#endif

	for(int iter = 0; iter < cgIter; iter++){
		//ap=A*p;
		//WARN: set temp to zero since the next operation is +=!
		for(int k = threadIdx.x; k < f; k += blockDim.x){
			temp = 0;
			for(int i = 0; i < f; i++)
				temp += A[blockIdx.x*f*f + f*i + k]*sharedp[i];
			sharedap[k] = temp;
		}
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("----------CG iteration %d \n", iter);
			printf("***ap:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
			printf("***shared memory content before 2rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(threadIdx.x == 0){
			rsnew[0] = 0;
		}
		//no need to have sync before blockReduce
		//because there is a __syncthreads() in blockReduce
		//pAp=p'*Ap
		temp = 0;
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			temp += sharedp[k]*sharedap[k];		
		//temp = blockReduceSum(shared, temp);
		blockReduceSumWithAtomics(rsnew, temp);
		//sync needed, to let all atomicAdd threads completes
		__syncthreads();
		if(threadIdx.x == 0){
			//pAp = temp;
			//alpha=rsold/(p'*Ap); use rsnew to store pAp
			alpha[0] = rsold[0]/rsnew[0];
			#ifdef DEBUG
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***pAp:\n");
			printf("pAp = %f \n", rsnew[0]);
			printf("***alpha:\n");
			printf("alpha = %f \n", alpha[0]);
			#endif
			rsnew[0] = 0;
		}
		//needed, aplpha[0] to be used by all threads
		__syncthreads();
		for(int k = threadIdx.x; k < f; k += blockDim.x){
			//x=x+alpha*p;
			sharedx[k] = 
				sharedx[k] + alpha[0] * sharedp[k];
			//r=r-alpha*Ap;
			sharedr[k] = 
				sharedr[k] - alpha[0] * sharedap[k];
		}
		//NOT needed?
		//__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content before 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		
		//rsnew=r'*r;
		/*
		temp = sharedr[threadIdx.x]*sharedr[threadIdx.x];
		temp = blockReduceSum(shared, temp);
		__syncthreads();
		if(threadIdx.x == 0){
			rsnew[0] = temp;
		}
		*/
		temp = 0;
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			temp += sharedr[k]*sharedr[k];
		blockReduceSumWithAtomics(rsnew, temp);
		//WARN: has to have this sync!
		__syncthreads();

		#ifdef DEBUG
		if(threadIdx.x==0){
			printf("***rsnew:\n");
			printf("rsnew = %f \n", rsnew[0]);
			printf("***shared memory content after 3rd blockReduceSum:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
		if(rsnew[0]<CG_ERROR)
			break;
		//NOT needed?
		//__syncthreads();
		//beta
		if(threadIdx.x == 0){
			beta[0] = rsnew[0]/rsold[0];
			//rsold=rsnew;
			rsold[0] = rsnew[0];
		}
		//need sync since every thread needs beta[0]
		__syncthreads();
		for(int k = threadIdx.x; k < f; k += blockDim.x)
			//p=r+(rsnew/rsold)*p;
			sharedp[k] = 
				sharedr[k] + beta[0] * sharedp[k];
		//need sync as every thread needs sharedp at the beginning of for
		__syncthreads();
		#ifdef DEBUG
		__syncthreads();
		if(threadIdx.x==0){
			printf("***shared memory content after update p:\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < f; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		__syncthreads();
		#endif
	}//end of CG iterations
	for(int k = threadIdx.x; k < f; k += blockDim.x)
		//x<--sharedx
		x[blockIdx.x*f + k] = sharedx[k];
}

void updateXWithCGHost_tt_fp16(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	updateXWithCGKernel3<<<batchSize, f, (4*f+4)*sizeof(float)>>>
		((half*)A, x, b, batchSize, f, cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	#ifdef DEBUG
	
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	float *A_fp32;
	cudacall(cudaMalloc((void** ) &A_fp32, f * f * sizeof(A_fp32[0])));
	fp16Array2fp32Array<<<(f*f-1)/1024 + 1, 1024>>>(A_fp32, (half*)A, f*f);
	cudaDeviceSynchronize();
	cudaCheckError();
	cudacall(cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	cudacall(cudaFree(A_fp32));
	
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	/*
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	*/
	#endif
}

void updateXWithCGHost(float * A, float * x, float * b, const int batchSize, const int f, const float cgIter){
	updateXWithCGKernel<<<batchSize, f, (4*f+4)*sizeof(float)>>>
	//updateXWithCGKernel2<<<batchSize, 96, (4*f+4)*sizeof(float)>>>
		(A, x, b, batchSize, f, cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();
	
	#ifdef DEBUG
	
	printf("***A[0]:\n");
	float *h_A = new float[f * f];
	float *A_fp32;
	cudacall(cudaMalloc((void** ) &A_fp32, f * f * sizeof(A_fp32[0])));
	fp16Array2fp32Array<<<(f*f-1)/1024 + 1, 1024>>>(A_fp32, (half*)A, f*f);
	cudaDeviceSynchronize();
	cudaCheckError();
	cudacall(cudaMemcpy(h_A, A_fp32, f * f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f*f; i++)
		printf("%f ", h_A[i]);
	printf("\n");
	delete [] h_A;
	cudacall(cudaFree(A_fp32));
	
	printf("***x[0]:\n");
	float *h_x = new float[f];
	cudacall(cudaMemcpy(h_x, x, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_x[i]);
	printf("\n");
	delete [] h_x;
	/*
	printf("***b[0]:\n");
	float *h_b = new float[f];
	cudacall(cudaMemcpy(h_b, b, f * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i = 0; i < f; i++)
		printf("%f ", h_b[i]);
	printf("\n");
	delete [] h_b;
	*/
	#endif
}		


//fused kernel, use thetaT to update XT
__global__ void
__launch_bounds__(64)
alsUpdateFeature100(const int batch_offset,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* thetaT, float* XT, float* ythetaT, int cgIter) {
	extern __shared__ float2 thetaTemp[];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

		int tile_x = 0;
		int tile_y = 0;

		int tile = F/10;
		for ( int i = 0; i < 10; i++){
			int end = ((20-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * tile;
				tile_y = (10 + threadIdx.x - end) * tile;
				break;
			}
		}
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			float2 theta;
			//copy texture --> smem, and sync

			//two layers: warp divergence unless we split at 32
			//require 32 >= SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				//int index = threadIdx.x;
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//for (int k = 50*(threadIdx.x/32); k < 50*(threadIdx.x/32) + 50; k += 2){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k+1]);
								thetaTemp[index * F/2 + k/2] = theta;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 51]);
								thetaTemp[index * F/2 + k/2 + 25] = theta;
							}
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[index*F/2], 0, F*sizeof(float));
				}
			}
			__syncthreads();

			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					accumulate_in_registers();
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();
		
		#ifdef DEBUG
		if(blockIdx.x==0 && threadIdx.x==0){
			printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
		}
		#endif
		
		//newly added CG phase
		//reuse the abundant shared memory
		float *sharedx = (float*)&thetaTemp[0];
		float *sharedp = (float*)&thetaTemp[50];
		float *sharedr = (float*)&thetaTemp[100];
		float *sharedap = (float*)&thetaTemp[150];
		float *sharedax = (float*)&thetaTemp[200];
		
		float *rsold = (float*)&thetaTemp[250]; 
		float *alpha = (float*)&thetaTemp[251];
		float *rsnew = (float*)&thetaTemp[252];
		float *beta = (float*)&thetaTemp[253];
		//sharedx<--x
		for(int k = threadIdx.x; k < F; k += 64){
			sharedx[k] = XT[blockIdx.x*F + k];
			sharedax[k] = 0;
		}
		__syncthreads();
		float temp = 0;
		//only uses 55 threads for A*p and A*x
		if(threadIdx.x < 55){
			//add regularization
			if(tile_x==tile_y){
				temp = (end - start) * lambda;
				temp0 += temp;
				temp11 += temp;
				temp22 += temp;
				temp33 += temp;
				temp44 += temp;
				temp55 += temp;
				temp66 += temp;
				temp77 += temp;
				temp88 += temp;
				temp99 += temp;
			}
			#ifdef DEBUG
			if(blockIdx.x==0 && threadIdx.x==0){
				printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
			}
			#endif
			//r=b-A*x;
			//step1: ax=A*x

			atomicAdd(&sharedax[tile_y], temp0*sharedx[tile_x] + temp10*sharedx[tile_x+1] + temp20*sharedx[tile_x+2] + temp30*sharedx[tile_x+3] +
					temp40*sharedx[tile_x + 4] + temp50*sharedx[tile_x + 5] + temp60*sharedx[tile_x + 6] + temp70*sharedx[tile_x + 7] +
					temp80*sharedx[tile_x + 8] + temp90*sharedx[tile_x + 9]);
					
			atomicAdd(&sharedax[tile_y+1], temp1*sharedx[tile_x] + temp11*sharedx[tile_x+1] + temp21*sharedx[tile_x+2] + temp31*sharedx[tile_x+3] + 
				temp41*sharedx[tile_x+4] + temp51*sharedx[tile_x+5] + temp61*sharedx[tile_x+6] +
				temp71*sharedx[tile_x+7] + temp81*sharedx[tile_x+8] + temp91*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+2], temp2*sharedx[tile_x] + temp12*sharedx[tile_x+1] + temp22*sharedx[tile_x+2] + temp32*sharedx[tile_x+3] + 
				temp42*sharedx[tile_x+4] + temp52*sharedx[tile_x+5] + temp62*sharedx[tile_x+6] +
				temp72*sharedx[tile_x+7] + temp82*sharedx[tile_x+8] + temp92*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+3], temp3*sharedx[tile_x] + temp13*sharedx[tile_x+1] + temp23*sharedx[tile_x+2] + temp33*sharedx[tile_x+3] + 
				temp43*sharedx[tile_x+4] + temp53*sharedx[tile_x+5] + temp63*sharedx[tile_x+6] +
				temp73*sharedx[tile_x+7] + temp83*sharedx[tile_x+8] + temp93*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+4], temp4*sharedx[tile_x] + temp14*sharedx[tile_x+1] + temp24*sharedx[tile_x+2] + temp34*sharedx[tile_x+3] + 
				temp44*sharedx[tile_x+4] + temp54*sharedx[tile_x+5] + temp64*sharedx[tile_x+6] +
				temp74*sharedx[tile_x+7] + temp84*sharedx[tile_x+8] + temp94*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+5], temp5*sharedx[tile_x] + temp15*sharedx[tile_x+1] + temp25*sharedx[tile_x+2] + temp35*sharedx[tile_x+3] + 
				temp45*sharedx[tile_x+4] + temp55*sharedx[tile_x+5] + temp65*sharedx[tile_x+6] +
				temp75*sharedx[tile_x+7] + temp85*sharedx[tile_x+8] + temp95*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+6], temp6*sharedx[tile_x] + temp16*sharedx[tile_x+1] + temp26*sharedx[tile_x+2] + temp36*sharedx[tile_x+3] + 
				temp46*sharedx[tile_x+4] + temp56*sharedx[tile_x+5] + temp66*sharedx[tile_x+6] +
				temp76*sharedx[tile_x+7] + temp86*sharedx[tile_x+8] + temp96*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+7], temp7*sharedx[tile_x] + temp17*sharedx[tile_x+1] + temp27*sharedx[tile_x+2] + temp37*sharedx[tile_x+3] + 
				temp47*sharedx[tile_x+4] + temp57*sharedx[tile_x+5] + temp67*sharedx[tile_x+6] +
				temp77*sharedx[tile_x+7] + temp87*sharedx[tile_x+8] + temp97*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+8], temp8*sharedx[tile_x] + temp18*sharedx[tile_x+1] + temp28*sharedx[tile_x+2] + temp38*sharedx[tile_x+3] + 
				temp48*sharedx[tile_x+4] + temp58*sharedx[tile_x+5] + temp68*sharedx[tile_x+6] +
				temp78*sharedx[tile_x+7] + temp88*sharedx[tile_x+8] + temp98*sharedx[tile_x+9]);
			atomicAdd(&sharedax[tile_y+9], temp9*sharedx[tile_x] + temp19*sharedx[tile_x+1] + temp29*sharedx[tile_x+2] + temp39*sharedx[tile_x+3] + 
				temp49*sharedx[tile_x+4] + temp59*sharedx[tile_x+5] + temp69*sharedx[tile_x+6] +
				temp79*sharedx[tile_x+7] + temp89*sharedx[tile_x+8] + temp99*sharedx[tile_x+9]);

			if(tile_x!=tile_y){
				atomicAdd(&sharedax[tile_x], temp0*sharedx[tile_y] + temp1*sharedx[tile_y + 1] + temp2*sharedx[tile_y + 2] + temp3*sharedx[tile_y + 3] +
					temp4*sharedx[tile_y + 4] + temp5*sharedx[tile_y + 5] + temp6*sharedx[tile_y + 6] + temp7*sharedx[tile_y + 7] +
					temp8*sharedx[tile_y + 8] + temp9*sharedx[tile_y + 9]);
					
				atomicAdd(&sharedax[tile_x+1], temp10*sharedx[tile_y] + temp11*sharedx[tile_y+1] + temp12*sharedx[tile_y+2] + temp13*sharedx[tile_y+3] + 
					temp14*sharedx[tile_y+4] + temp15*sharedx[tile_y+5] + temp16*sharedx[tile_y+6] +
					temp17*sharedx[tile_y+7] + temp18*sharedx[tile_y+8] + temp19*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+2], temp20*sharedx[tile_y] + temp21*sharedx[tile_y+1] + temp22*sharedx[tile_y+2] + temp23*sharedx[tile_y+3] + 
					temp24*sharedx[tile_y+4] + temp25*sharedx[tile_y+5] + temp26*sharedx[tile_y+6] +
					temp27*sharedx[tile_y+7] + temp28*sharedx[tile_y+8] + temp29*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+3], temp30*sharedx[tile_y] + temp31*sharedx[tile_y+1] + temp32*sharedx[tile_y+2] + temp33*sharedx[tile_y+3] + 
					temp34*sharedx[tile_y+4] + temp35*sharedx[tile_y+5] + temp36*sharedx[tile_y+6] +
					temp37*sharedx[tile_y+7] + temp38*sharedx[tile_y+8] + temp39*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+4], temp40*sharedx[tile_y] + temp41*sharedx[tile_y+1] + temp42*sharedx[tile_y+2] + temp43*sharedx[tile_y+3] + 
					temp44*sharedx[tile_y+4] + temp45*sharedx[tile_y+5] + temp46*sharedx[tile_y+6] +
					temp47*sharedx[tile_y+7] + temp48*sharedx[tile_y+8] + temp49*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+5], temp50*sharedx[tile_y] + temp51*sharedx[tile_y+1] + temp52*sharedx[tile_y+2] + temp53*sharedx[tile_y+3] + 
					temp54*sharedx[tile_y+4] + temp55*sharedx[tile_y+5] + temp56*sharedx[tile_y+6] +
					temp57*sharedx[tile_y+7] + temp58*sharedx[tile_y+8] + temp59*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+6], temp60*sharedx[tile_y] + temp61*sharedx[tile_y+1] + temp62*sharedx[tile_y+2] + temp63*sharedx[tile_y+3] + 
					temp64*sharedx[tile_y+4] + temp65*sharedx[tile_y+5] + temp66*sharedx[tile_y+6] +
					temp67*sharedx[tile_y+7] + temp68*sharedx[tile_y+8] + temp69*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+7], temp70*sharedx[tile_y] + temp71*sharedx[tile_y+1] + temp72*sharedx[tile_y+2] + temp73*sharedx[tile_y+3] + 
					temp74*sharedx[tile_y+4] + temp75*sharedx[tile_y+5] + temp76*sharedx[tile_y+6] +
					temp77*sharedx[tile_y+7] + temp78*sharedx[tile_y+8] + temp79*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+8], temp80*sharedx[tile_y] + temp81*sharedx[tile_y+1] + temp82*sharedx[tile_y+2] + temp83*sharedx[tile_y+3] + 
					temp84*sharedx[tile_y+4] + temp85*sharedx[tile_y+5] + temp86*sharedx[tile_y+6] +
					temp87*sharedx[tile_y+7] + temp88*sharedx[tile_y+8] + temp89*sharedx[tile_y+9]);
				atomicAdd(&sharedax[tile_x+9], temp90*sharedx[tile_y] + temp91*sharedx[tile_y+1] + temp92*sharedx[tile_y+2] + temp93*sharedx[tile_y+3] + 
					temp94*sharedx[tile_y+4] + temp95*sharedx[tile_y+5] + temp96*sharedx[tile_y+6] +
					temp97*sharedx[tile_y+7] + temp98*sharedx[tile_y+8] + temp99*sharedx[tile_y+9]);
			}

		}
		__syncthreads();

		#ifdef DEBUG
		if(blockIdx.x==0 && threadIdx.x==0){
			printf("***x:\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedx[i]);
			printf("\n\n");
			printf("***r=Ax:\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedax[i]);
			printf("\n\n");
		}
		#endif
		for(int k = threadIdx.x; k < F; k += 64){
			//r=b-Ax
			sharedr[k] = ythetaT[blockIdx.x*blockDim.x + k] - sharedax[k];
			//p=r;
			sharedp[k] = sharedr[k];
		}
		//rsold=r'*r;
		if(threadIdx.x == 0){
			rsold[0] = 0;
		}
		for(int k = threadIdx.x; k < F; k += 64){
			temp += sharedr[k]*sharedr[k];
		}
		blockReduceSumWithAtomics(rsold, temp);	
		__syncthreads();
		#ifdef DEBUG
		if(blockIdx.x==0 && threadIdx.x==0){
			printf("***rsold:\n");
			printf("rsold = %f \n", rsold[0]);
			printf("***shared memory content after 1st blockReduceSum:\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedx[i]);
			printf("\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedax[i]);
			printf("\n\n");

			for(int i = 0; i < 100; i++)
				printf("%f ", sharedp[i]);
			printf("\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedr[i]);
			printf("\n");
			for(int i = 0; i < 100; i++)
				printf("%f ", sharedap[i]);
			printf("\n");
		}
		#endif
///*
		//CG iterations
		for(int iter = 0; iter < cgIter; iter++){
			//ap=A*p;
			for(int k = threadIdx.x; k < F; k += 64)
				sharedap[k] = 0;
			__syncthreads();
			//only uses 55 threads for A*p and A*x
			if(threadIdx.x < 55){
				atomicAdd(&sharedap[tile_y], temp0*sharedp[tile_x] + temp10*sharedp[tile_x+1] + temp20*sharedp[tile_x+2] + temp30*sharedp[tile_x+3] +
					temp40*sharedp[tile_x + 4] + temp50*sharedp[tile_x + 5] + temp60*sharedp[tile_x + 6] + temp70*sharedp[tile_x + 7] +
					temp80*sharedp[tile_x + 8] + temp90*sharedp[tile_x + 9]);
						
				atomicAdd(&sharedap[tile_y+1], temp1*sharedp[tile_x] + temp11*sharedp[tile_x+1] + temp21*sharedp[tile_x+2] + temp31*sharedp[tile_x+3] + 
					temp41*sharedp[tile_x+4] + temp51*sharedp[tile_x+5] + temp61*sharedp[tile_x+6] +
					temp71*sharedp[tile_x+7] + temp81*sharedp[tile_x+8] + temp91*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+2], temp2*sharedp[tile_x] + temp12*sharedp[tile_x+1] + temp22*sharedp[tile_x+2] + temp32*sharedp[tile_x+3] + 
					temp42*sharedp[tile_x+4] + temp52*sharedp[tile_x+5] + temp62*sharedp[tile_x+6] +
					temp72*sharedp[tile_x+7] + temp82*sharedp[tile_x+8] + temp92*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+3], temp3*sharedp[tile_x] + temp13*sharedp[tile_x+1] + temp23*sharedp[tile_x+2] + temp33*sharedp[tile_x+3] + 
					temp43*sharedp[tile_x+4] + temp53*sharedp[tile_x+5] + temp63*sharedp[tile_x+6] +
					temp73*sharedp[tile_x+7] + temp83*sharedp[tile_x+8] + temp93*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+4], temp4*sharedp[tile_x] + temp14*sharedp[tile_x+1] + temp24*sharedp[tile_x+2] + temp34*sharedp[tile_x+3] + 
					temp44*sharedp[tile_x+4] + temp54*sharedp[tile_x+5] + temp64*sharedp[tile_x+6] +
					temp74*sharedp[tile_x+7] + temp84*sharedp[tile_x+8] + temp94*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+5], temp5*sharedp[tile_x] + temp15*sharedp[tile_x+1] + temp25*sharedp[tile_x+2] + temp35*sharedp[tile_x+3] + 
					temp45*sharedp[tile_x+4] + temp55*sharedp[tile_x+5] + temp65*sharedp[tile_x+6] +
					temp75*sharedp[tile_x+7] + temp85*sharedp[tile_x+8] + temp95*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+6], temp6*sharedp[tile_x] + temp16*sharedp[tile_x+1] + temp26*sharedp[tile_x+2] + temp36*sharedp[tile_x+3] + 
					temp46*sharedp[tile_x+4] + temp56*sharedp[tile_x+5] + temp66*sharedp[tile_x+6] +
					temp76*sharedp[tile_x+7] + temp86*sharedp[tile_x+8] + temp96*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+7], temp7*sharedp[tile_x] + temp17*sharedp[tile_x+1] + temp27*sharedp[tile_x+2] + temp37*sharedp[tile_x+3] + 
					temp47*sharedp[tile_x+4] + temp57*sharedp[tile_x+5] + temp67*sharedp[tile_x+6] +
					temp77*sharedp[tile_x+7] + temp87*sharedp[tile_x+8] + temp97*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+8], temp8*sharedp[tile_x] + temp18*sharedp[tile_x+1] + temp28*sharedp[tile_x+2] + temp38*sharedp[tile_x+3] + 
					temp48*sharedp[tile_x+4] + temp58*sharedp[tile_x+5] + temp68*sharedp[tile_x+6] +
					temp78*sharedp[tile_x+7] + temp88*sharedp[tile_x+8] + temp98*sharedp[tile_x+9]);
				atomicAdd(&sharedap[tile_y+9], temp9*sharedp[tile_x] + temp19*sharedp[tile_x+1] + temp29*sharedp[tile_x+2] + temp39*sharedp[tile_x+3] + 
					temp49*sharedp[tile_x+4] + temp59*sharedp[tile_x+5] + temp69*sharedp[tile_x+6] +
					temp79*sharedp[tile_x+7] + temp89*sharedp[tile_x+8] + temp99*sharedp[tile_x+9]);

				if(tile_x!=tile_y){
					atomicAdd(&sharedap[tile_x], temp0*sharedp[tile_y] + temp1*sharedp[tile_y + 1] + temp2*sharedp[tile_y + 2] + temp3*sharedp[tile_y + 3] +
						temp4*sharedp[tile_y + 4] + temp5*sharedp[tile_y + 5] + temp6*sharedp[tile_y + 6] + temp7*sharedp[tile_y + 7] +
						temp8*sharedp[tile_y + 8] + temp9*sharedp[tile_y + 9]);
						
					atomicAdd(&sharedap[tile_x+1], temp10*sharedp[tile_y] + temp11*sharedp[tile_y+1] + temp12*sharedp[tile_y+2] + temp13*sharedp[tile_y+3] + 
						temp14*sharedp[tile_y+4] + temp15*sharedp[tile_y+5] + temp16*sharedp[tile_y+6] +
						temp17*sharedp[tile_y+7] + temp18*sharedp[tile_y+8] + temp19*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+2], temp20*sharedp[tile_y] + temp21*sharedp[tile_y+1] + temp22*sharedp[tile_y+2] + temp23*sharedp[tile_y+3] + 
						temp24*sharedp[tile_y+4] + temp25*sharedp[tile_y+5] + temp26*sharedp[tile_y+6] +
						temp27*sharedp[tile_y+7] + temp28*sharedp[tile_y+8] + temp29*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+3], temp30*sharedp[tile_y] + temp31*sharedp[tile_y+1] + temp32*sharedp[tile_y+2] + temp33*sharedp[tile_y+3] + 
						temp34*sharedp[tile_y+4] + temp35*sharedp[tile_y+5] + temp36*sharedp[tile_y+6] +
						temp37*sharedp[tile_y+7] + temp38*sharedp[tile_y+8] + temp39*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+4], temp40*sharedp[tile_y] + temp41*sharedp[tile_y+1] + temp42*sharedp[tile_y+2] + temp43*sharedp[tile_y+3] + 
						temp44*sharedp[tile_y+4] + temp45*sharedp[tile_y+5] + temp46*sharedp[tile_y+6] +
						temp47*sharedp[tile_y+7] + temp48*sharedp[tile_y+8] + temp49*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+5], temp50*sharedp[tile_y] + temp51*sharedp[tile_y+1] + temp52*sharedp[tile_y+2] + temp53*sharedp[tile_y+3] + 
						temp54*sharedp[tile_y+4] + temp55*sharedp[tile_y+5] + temp56*sharedp[tile_y+6] +
						temp57*sharedp[tile_y+7] + temp58*sharedp[tile_y+8] + temp59*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+6], temp60*sharedp[tile_y] + temp61*sharedp[tile_y+1] + temp62*sharedp[tile_y+2] + temp63*sharedp[tile_y+3] + 
						temp64*sharedp[tile_y+4] + temp65*sharedp[tile_y+5] + temp66*sharedp[tile_y+6] +
						temp67*sharedp[tile_y+7] + temp68*sharedp[tile_y+8] + temp69*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+7], temp70*sharedp[tile_y] + temp71*sharedp[tile_y+1] + temp72*sharedp[tile_y+2] + temp73*sharedp[tile_y+3] + 
						temp74*sharedp[tile_y+4] + temp75*sharedp[tile_y+5] + temp76*sharedp[tile_y+6] +
						temp77*sharedp[tile_y+7] + temp78*sharedp[tile_y+8] + temp79*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+8], temp80*sharedp[tile_y] + temp81*sharedp[tile_y+1] + temp82*sharedp[tile_y+2] + temp83*sharedp[tile_y+3] + 
						temp84*sharedp[tile_y+4] + temp85*sharedp[tile_y+5] + temp86*sharedp[tile_y+6] +
						temp87*sharedp[tile_y+7] + temp88*sharedp[tile_y+8] + temp89*sharedp[tile_y+9]);
					atomicAdd(&sharedap[tile_x+9], temp90*sharedp[tile_y] + temp91*sharedp[tile_y+1] + temp92*sharedp[tile_y+2] + temp93*sharedp[tile_y+3] + 
						temp94*sharedp[tile_y+4] + temp95*sharedp[tile_y+5] + temp96*sharedp[tile_y+6] +
						temp97*sharedp[tile_y+7] + temp98*sharedp[tile_y+8] + temp99*sharedp[tile_y+9]);
				}
			}
			__syncthreads();
			#ifdef DEBUG
			if(blockIdx.x==0 && threadIdx.x==0){
				printf("----------CG iteration %d \n", iter);
				printf("***ap:\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedap[i]);
				printf("\n\n");
				printf("***shared memory content before 2rd blockReduceSum:\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedp[i]);
				printf("\n\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedr[i]);
				printf("\n\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedap[i]);
				printf("\n\n");
			}
			#endif
			if(threadIdx.x == 0){
				rsnew[0] = 0;
			}
			//no need to have sync before blockReduce
			//because there is a __syncthreads() in blockReduce
			//pAp=p'*Ap
			temp = 0;
			for(int k = threadIdx.x; k < F; k += 64)
				temp += sharedp[k]*sharedap[k];		
			//temp = blockReduceSum(shared, temp);
			blockReduceSumWithAtomics(rsnew, temp);
			//sync needed, to let all atomicAdd threads completes
			__syncthreads();
			if(threadIdx.x == 0){
				//pAp = temp;
				//alpha=rsold/(p'*Ap); use rsnew to store pAp
				alpha[0] = rsold[0]/rsnew[0];
				#ifdef DEBUG
				if(blockIdx.x==0){
					printf("***rsold:\n");
					printf("rsold = %f \n", rsold[0]);
					printf("***pAp:\n");
					printf("pAp = %f \n", rsnew[0]);
					printf("***alpha:\n");
					printf("alpha = %f \n", alpha[0]);
				}
				#endif
				rsnew[0] = 0;
			}
			//needed, aplpha[0] to be used by all threads
			__syncthreads();
			for(int k = threadIdx.x; k < F; k += 64){
				//x=x+alpha*p;
				sharedx[k] = sharedx[k] + alpha[0] * sharedp[k];
				//r=r-alpha*Ap;
				sharedr[k] = sharedr[k] - alpha[0] * sharedap[k];
				//NOT needed?
				//__syncthreads();
			}
			__syncthreads();
			#ifdef DEBUG
			if(blockIdx.x==0 && threadIdx.x==0){
				printf("***shared memory content before 3rd blockReduceSum:\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedp[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedr[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedap[i]);
				printf("\n");
			}
			#endif		
			//rsnew=r'*r;
			temp = 0;
			for(int k = threadIdx.x; k < F; k += 64)
				temp += sharedr[k]*sharedr[k];
			blockReduceSumWithAtomics(rsnew, temp);
			//WARN: has to have this sync!
			__syncthreads();

			#ifdef DEBUG
			if(blockIdx.x==0 && threadIdx.x==0){
				printf("***rsnew:\n");
				printf("rsnew = %f \n", rsnew[0]);
				printf("***shared memory content after 3rd blockReduceSum:\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedp[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedr[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedap[i]);
				printf("\n");
			}
			#endif
			if(rsnew[0]<CG_ERROR)
				break;
			//NOT needed?
			//__syncthreads();
			//beta
			if(threadIdx.x == 0){
				beta[0] = rsnew[0]/rsold[0];
				//rsold=rsnew;
				rsold[0] = rsnew[0];
			}
			//need sync since every thread needs beta[0]
			__syncthreads();
			//p=r+(rsnew/rsold)*p;
			for(int k = threadIdx.x; k < F; k += 64)
				sharedp[k] = sharedr[k] + beta[0] * sharedp[k];
			//need sync as every thread needs sharedp at the beginning of for
			__syncthreads();
			#ifdef DEBUG
			__syncthreads();
			if(blockIdx.x==0 && threadIdx.x==0){
				printf("***shared memory content after update p:\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedp[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedr[i]);
				printf("\n");
				for(int i = 0; i < F; i++)
					printf("%f ", sharedap[i]);
				printf("\n");
			}
			__syncthreads();
			#endif
		}//end of CG iterations
		//x<--sharedx
		for(int k = threadIdx.x; k < F; k += 64)
			XT[blockIdx.x*F + k] = sharedx[k];
//*/		
	}
}
void alsUpdateFeature100Host(const int batch_offset,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* __restrict__ thetaT, float* XT, float* ythetaT, int cgIter){
	alsUpdateFeature100<<<m, 64, SCAN_BATCH * F/2*sizeof(float2)>>>
					(batch_offset, csrRowIndex, csrColIndex, lambda, m, F, thetaT, XT, ythetaT, cgIter);
	cudaDeviceSynchronize();
	cudaCheckError();		
}