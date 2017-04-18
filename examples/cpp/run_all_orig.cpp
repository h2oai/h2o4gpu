#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "examples.h"

typedef float real_t;

int main(int argc, char **argv) {
  double t;

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


  std::ifstream ifs("./train.txt");
  std::string line;
  int rows=0;
  int cols = 0;
  while (std::getline(ifs, line)) {
    if (rows==0) {
      std::string buf;
      std::stringstream ss(line);
      while (ss >> buf) cols++;
    }
    //std::cout << line << std::endl;
    rows++;
  }
  cols--; //don't count target column

  printf("rows: %d\n", rows); fflush(stdout);
  printf("cols (w/o response): %d\n", cols); fflush(stdout);
  ifs.close();

  printf("\nLasso: rows=%d n=%d.\n",rows,cols);
  t = Lasso<real_t>(rows, cols);
  printf("Lasso rows=%d n=%d Solver Time: %e sec\n", rows,cols,t);

  
  printf("\nLassoPath: rows=%d n=%d.\n",rows,cols);
  t = LassoPath<real_t>(rows, cols);
  printf("LassoPath rows=%d n=%d Solver Time: %e sec\n", rows,cols,t);

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

