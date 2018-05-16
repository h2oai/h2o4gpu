/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include <string>
#include "timer.h"

Timer::Timer() {
  reset();
}

void Timer::reset() {
  begin = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(begin - begin);
}

void Timer::tic() {
  begin = std::chrono::high_resolution_clock::now();
}

float Timer::toc() {
  duration += std::chrono::duration_cast<std::chrono::milliseconds>
      (std::chrono::high_resolution_clock::now() - begin);
  return peek();
}

float Timer::peek() {
  return (float) duration.count() / 1000;
}

float Timer::pop() {
  float secs = (float) duration.count() / 1000;
  this->reset();
  return secs;
}