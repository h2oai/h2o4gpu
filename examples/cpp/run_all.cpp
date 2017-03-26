#include <cstdio>

#include "examples.h"

typedef float real_t;

int main() {
  double t;

  /*
  printf("\nLogistic Regression.\n");
  t = Logistic<real_t>(7000, 100000);
  printf("Solver Time: %e sec\n", t);
  */
 

  printf("\nLasso.\n");
  t = Lasso<real_t>(100000, 2000);
  printf("Solver Time: %e sec\n", t);

  printf("\nLasso Path.\n");
  t = LassoPath<real_t>(100000, 2000);
  printf("Solver Time: %e sec\n", t);
  /*
  printf("\nLinear Program in Equality Form.\n");
  t = LpEq<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);

  printf("\nLinear Program in Inequality Form.\n");
  t = LpIneq<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);

  printf("\nNon-Negative Least Squares.\n");
  t = NonNegL2<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);

  printf("\nSupport Vector Machine.\n");
  t = Svm<real_t>(1000, 200);
  printf("Solver Time: %e sec\n", t);
*/

  return 0;
}

