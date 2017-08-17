#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "examples.h"
#include "../reader.h"
#include "timer.h"

typedef float real_t;

int main(int argc, char **argv) {
  double t;
  using namespace std;

  /*
  printf("\nLogistic Regression.\n");
  t = Logistic<real_t>(7000, 100000);
  printf("Solver Time: %e sec\n", t);
  */

  //  int rows=450016; // BARELY succeeds in 2000 iterations with cols=213 and no adpative rho.  Fails with adaptive rho.  No big changes with Eq constant changes.

  //int rows=5000;
  //  int rows=450000; // testing, succeeds with adaptive rho, but fails without adaptive rho
  //  int rows=450020; // testing, fails with or without adaptive rho

  
  //  int  rows=450000; // fails without adaptive rho, works perfectly fine with adaptive rho.
  //  int rows=450050;
  //  int cols=212;

  //  extern int bwcheck(void);
  //  bwcheck();
  //  return(0);

  size_t rows=0, cols=0;

  std::vector<real_t> A;
  std::vector<real_t> b;
  std::vector<real_t> w;
  cout << "START FILL DATA\n" << endl;
  double t0 = timer<double>();
  fillData(rows,cols,"train.txt", A, b, w);
  double t1 = timer<double>();
  rows=b.size();
  cols=A.size()/b.size();
  cout << "END FILL DATA. Took " << t1 - t0 << " secs" << endl;
  printf("rows: %zu\n", rows); fflush(stdout);
  printf("cols (w/o response): %zu\n", cols); fflush(stdout);

  printf("\nLasso: rows=%zu n=%zu.\n",rows,cols);
  t = Lasso<real_t>(A,b);
  printf("Lasso rows=%zu n=%zu Solver Time: %e sec\n", rows,cols,t);

  
  printf("\nLassoPath: rows=%zu n=%zu.\n",rows,cols);
  t = LassoPath<real_t>(A,b);
  printf("LassoPath rows=%zu n=%zu Solver Time: %e sec\n", rows,cols,t);

  fflush(stdout);
  fflush(stderr);
  

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

