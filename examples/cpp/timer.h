#ifndef TIMER_HPP_ 
#define TIMER_HPP_ 

#include <unistd.h>
#include <sys/time.h>

template <typename T>
T timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<T>(tv.tv_sec) +
      static_cast<T>(tv.tv_usec) / static_cast<T>(1e6);
}

#endif /* TIMER_HPP_ */

