#include <stdlib.h>
#include <cmath>
#include <cgls.cu>

typedef double real_t;

// Generates random CSR matrix with entries in [-1, 1]. The matrix will have 
// exactly nnz non-zeros. All arrays must be pre-allocated.
void CsrMatGen(int m, int n, int nnz, real_t *val, int *row_ptr, int *col_ind) {
  real_t kRandMax = static_cast<real_t>(RAND_MAX);

  int num = 0;
  for (int i = 0; i < m; ++i) {
    row_ptr[i] = num;
    for (int j = 0; j < n; ++j) {
      if (rand() / kRandMax * ((m - i) * n - j) < (nnz - num)) {
        val[num] = 2 * (rand() - kRandMax / 2) / kRandMax;
        col_ind[num] = j;
        num++;
      }
    }
  }
  row_ptr[m] = nnz;
}

// Test CGLS on small system of equations with known solution.
void test1() {
  // Initialize variables.
  real_t shift = 1;
  real_t tol = 1e-6;
  int maxit = 20;
  bool quiet = false;
  int m = 5;
  int n = 5;
  int nnz = 13;
  
  // Initialize data.
  real_t val_h[]    = { 1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5 }; 
  int col_ind_h[]   = { 0,  1,  3,  0,  1,  2,  3,  4,  0,  2,  3,  1,  4 };
  int row_ptr_h[]   = { 0,  3,  5,  8, 11, 13 };
  real_t b_h[]      = {-2, -1,  0,  1,  2 };
  real_t x_h[]      = { 0,  0,  0,  0,  0 };
  real_t x_star[] = { 0.461620337853983,
                      0.025458521291462,
                     -0.509793131412600,
                      0.579159637092979,
                     -0.350590484189795 };

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *col_ind_d, *row_ptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&col_ind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  row_ptr_d = col_ind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(row_ptr_d, row_ptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve. 
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, row_ptr_d, col_ind_d, m, n,
      nnz, b_d, x_d, shift, tol, maxit, quiet);
  
  // Retrieve solution
  cudaMemcpy(x_h, x_d, n * sizeof(real_t), cudaMemcpyDeviceToHost);

  // Compute error and print.
  real_t err = 0;
  for (int i = 0; i < m; ++i)
    err += (x_h[i] - x_star[i]) * (x_h[i] - x_star[i]);
  err = std::sqrt(err);

  printf("Flag = %d, Error = %e\n", flag, err);
  cudaFree(val_d);
  cudaFree(col_ind_d);
}

// Test CGLS on larger random matrix.
void test2() {
  // Initialize variables.
  real_t shift = 1;
  real_t tol = 1e-6;
  int maxit = 20;
  bool quiet = false;
  int m = 100;
  int n = 1000;
    int nnz = 5000;

  // Initialize data.
  real_t val_h[nnz];
  int col_ind_h[nnz];
  int row_ptr_h[m + 1];
  real_t b_h[m];
  real_t x_h[n];
  
  // Generate data.
  CsrMatGen(m, n, nnz, val_h, row_ptr_h, col_ind_h);
  for (int i = 0; i < m; ++i)
    b_h[i] = rand() / static_cast<real_t>(RAND_MAX);
  for (int i = 0; i < n; ++i)
    x_h[i] = 0;

  // Transfer variables to device.
  real_t *val_d, *b_d, *x_d;
  int *col_ind_d, *row_ptr_d;

  cudaMalloc(&val_d, (nnz + m + n) * sizeof(real_t));
  cudaMalloc(&col_ind_d, (nnz + m + 1) * sizeof(int));
  b_d = val_d + nnz;
  x_d = b_d + m;
  row_ptr_d = col_ind_d + nnz;

  cudaMemcpy(val_d, val_h, nnz * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, m * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(row_ptr_d, row_ptr_h, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Solve. 
  int flag = cgls::solve<real_t, cgls::CSR>(val_d, row_ptr_d, col_ind_d, m, n,
      nnz, b_d, x_d, shift, tol, maxit, quiet);
  
  // Retrieve solution
  cudaMemcpy(x_h, x_d, n * sizeof(real_t), cudaMemcpyDeviceToHost);

  printf("Flag = %d\n", flag);
  cudaFree(val_d);
  cudaFree(col_ind_d);
}

// Run tests.
int main() {
  test1();
  test2();
}

