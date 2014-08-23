#ifndef MP1_UTIL_H_
#define MP1_UTIL_H_

#include <iostream>

struct event_pair {
  cudaEvent_t start;
  cudaEvent_t end;
};


inline void check_launch(char * kernel_name) {
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}


inline void start_timer(event_pair * p) {
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair * p) {
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

union Float_t {
  Float_t(float num) : f(num) {}
  bool Negative() const { return (i >> 31) != 0; }
  int32_t i;
  float f;
};
 
// ref: http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
bool AlmostEqualUlps(float A, float B, int maxUlpsDiff) {
  Float_t uA(A);
  Float_t uB(B);

  // Different signs means they do not match.
  if (uA.Negative() != uB.Negative()) {
    // Check for equality to make sure +0==-0
    if (A == B)
        return true;
    return false;
  }

  // Find the difference in ULPs.
  int ulpsDiff = abs(uA.i - uB.i);
  if (ulpsDiff <= maxUlpsDiff)
    return true;

  return false;
}

#endif /* MP1_UTIL_H_ */

