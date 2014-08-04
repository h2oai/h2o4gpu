#include <cstdio>

#include "examples.h"

typedef double real_t;

int main() {
  real_t t;
  printf("\nLasso.\n");
  t = Lasso<real_t>(1000, 10000);
  printf("Solver Time: %e sec\n", t);

  printf("\nLinear Program in Equality Form.\n");
  t = LpEq<real_t>(200, 1000);
  printf("Solver Time: %e sec\n", t);

  printf("\nLinear Program in Inequality Form.\n");
  t = LpIneq<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);

  printf("\nNon-Negative Least Squares.\n");
  t = NonNegL2<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);

  printf("\nSupport Vector Machine.\n");
  t = Svm<real_t>(1000, 100);
  printf("Solver Time: %e sec\n", t);

  return 0;
}

