#ifndef TIMER_H_ 
#define TIMER_H_ 

#include <unistd.h>
#include <sys/time.h>

template <typename T>
T timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<T>(tv.tv_sec) +
      static_cast<T>(tv.tv_usec) / static_cast<T>(1e6);
}

#endif  // TIMER_H_

