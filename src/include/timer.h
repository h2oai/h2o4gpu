/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
#ifndef TIMER_H_
#define TIMER_H_

#ifdef _WIN32
#include <chrono>

template <typename T>
T timer() {
  return std::chrono::duration<T>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
#else
#include <unistd.h>
#include <sys/time.h>

template <typename T>
T timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<T>(tv.tv_sec) +
      static_cast<T>(tv.tv_usec) * static_cast<T>(1e-6);
}
#endif

#endif  // UTIL_H_