#include <cstdio>

#include "examples.h"

typedef double real_t;

int main() {
//  double t;
//  printf("\nLasso.\n");
//  t = LpEq<real_t>(1000, 200);
//  printf("Solver Time: %e sec\n", t);

  LpEq<float >(1000, 200);
  LpEq<double>(1000, 200);

  return 0;
}

