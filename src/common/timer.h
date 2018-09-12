/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <chrono>

class Timer {
 public:

  Timer();

  void reset();
  void tic();
  float toc();
  float peek();
  float pop();

 private:
  std::chrono::high_resolution_clock::time_point begin;
  std::chrono::milliseconds duration;
};
