#include <cstdio>
#include <math.h>

#include "examples.h"

typedef float real_t;

int main() {
  double t;

  /*
  printf("\nLogistic Regression.\n");
  t = Logistic<real_t>(7000, 100000);
  printf("Solver Time: %e sec\n", t);
  */

  //  int m=450016; // BARELY succeeds in 2000 iterations with n=213 and no adpative rho.  Fails with adaptive rho.  No big changes with Eq constant changes.


  //  int m=450000; // testing, succeeds with or without adaptive rho
  int m=450020; // testing, fails with or without adaptive rho

  
  //  int  m=450000; // fails without adaptive rho, works perfectly fine with adaptive rho.
  //  int m=450050;
  //  int n=212;
  int n=213;

  printf("\nLasso: m=%d n=%d.\n",m,n);
  t = Lasso<real_t>(m, n);
  printf("Lasso m=%d n=%d Solver Time: %e sec\n", m,n,t);

  /*
  printf("\nLasso Path: m=%d n=%d.\n",m,n);
  t = LassoPath<real_t>(m, n);
  printf("LassoPath m=%d n=%d Solver Time: %e sec\n", m,n,t);
  fflush(stdout);
  fflush(stderr);
  */

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

