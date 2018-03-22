/*!
 * Modifications Copyright 2017 H2O.ai, Inc.
 */
#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include "cuda_utils2.h"

inline int checkwDev(int wDev){
#ifdef DEBUG
  int nVis = 0;
#pragma omp critical
  {
  CUDACHECK(cudaGetDeviceCount(&nVis));
  }
  #ifdef DEBUG
  for (int i = 0; i < nVis; i++){
    cudaDeviceProp props;
    CUDACHECK(cudaGetDeviceProperties(&props, i));
    printf("Visible: Compute %d.%d CUDA device: [%s] : cudadeviceid: %2d of %2d devices [0x%02x] mpc=%d\n", props.major, props.minor, props.name, i\
           , nVis, props.pciBusID, props.multiProcessorCount); fflush(stdout);
  }
  #endif
  if(wDev>nVis-1){
    fprintf(stderr,"Not enough GPUs, where wDev=%d and nVis=%d\n",wDev,nVis);
    exit(1);
    return(1);
  }
  else return(0);
#else
return(0);
#endif
}


#endif
