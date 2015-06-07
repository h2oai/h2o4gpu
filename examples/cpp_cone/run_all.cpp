#include <cstdio>

#include "examples.h"

typedef float real_t;

int main() {
  double t;
  printf("\nLasso.\n");
  t = LpEq<real_t>(200, 2000);
  printf("Solver Time: %e sec\n", t);

  return 0;
}

