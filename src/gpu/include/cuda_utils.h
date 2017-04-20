#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Cuda failure %s:%d '%s'\n",             \
             __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)


int checkwDev(int wDev){
  int nVis = 0;
#pragma omp critical
  {
  CUDACHECK(cudaGetDeviceCount(&nVis));
  }
  #ifdef _DEBUG
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
}


#endif
