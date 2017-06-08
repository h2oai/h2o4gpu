#pragma once
#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
#endif

#include <sstream>
#include <stdio.h>
#include "timer.h"

namespace h2oaikmeans {

  static const std::string H2OAIKMEANS_VERSION = "0.0.1";

  template <typename M>
  class H2OAIKMeans {
    private:
      // Data
      const M* _A;
      int _k;
      size_t _n;
      size_t _d;
    public:
      H2OAIKMeans(const M *A, int k, size_t n, size_t d);
      int Solve();
  };

}  // namespace h2oaikmeans
