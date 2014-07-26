#ifndef SINKHORN_KNOPP_H_
#define SINKHORN_KNOPP_H_

#include <cmath>

#include "gsl/gsl_blas.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

// Sinkhorn Knopp algorithm for matrix equilibration.
// The following approx. holds: diag(d) * Ain * e =  1, diag(e) * Ain' * d = 1
// Output matrix is generated as: Aout = diag(d) * Ain * diag(e),
// template <typename T>
// void SinkhornKnopp(const gsl::matrix<T> *Ain, gsl::matrix<T> *Aout,
//                    gsl::vector<T> *d, gsl::vector<T> *e) {
//   unsigned int kNumItr = 10;
//   const T kOne = static_cast<T>(1);
//   const T kZero = static_cast<T>(0);
//   gsl::matrix_memcpy(Aout, Ain);
//   gsl::vector_set_all(d, kOne);
// 
//   // A := |A| -- elementwise
//   for (unsigned int i = 0; i < Aout->size1; ++i)
//     for (unsigned int j = 0; j < Aout->size2; ++j)
//       gsl::matrix_set(Aout, i, j, std::fabs(gsl::matrix_get(Aout, i, j)));
// 
//   // e := 1 ./ A' * d; d := 1 ./ A * e; -- k times.
//   for (unsigned int k = 0; k < kNumItr; ++k) {
//     gsl::blas_gemv(CblasTrans, kOne, Aout, d, kZero, e);
//     for (unsigned int i = 0; i < e->size; ++i)
//       gsl::vector_set(e, i, kOne / gsl::vector_get(e, i));
// 
//     gsl::blas_gemv(CblasNoTrans, kOne, Aout, e, kZero, d);
//     for (unsigned int i = 0; i < d->size; ++i)
//       gsl::vector_set(d, i, kOne / gsl::vector_get(d, i));
// 
//     T nrm_d = gsl::blas_nrm2(d) / std::sqrt(static_cast<T>(d->size));
//     T nrm_e = gsl::blas_nrm2(e) / std::sqrt(static_cast<T>(e->size));
//     T scale = std::sqrt(nrm_e / nrm_d);
//     gsl::blas_scal(scale, d);
//     gsl::blas_scal(kOne / scale, e);
//   }
// 
//   // A := D * A * E
//   gsl::matrix_memcpy(Aout, Ain);
//   for (unsigned int i = 0; i < Ain->size1; ++i) {
//     gsl::vector<T> v = gsl::matrix_row(Aout, i);
//     gsl::blas_scal(gsl::vector_get(d, i), &v);
//   }
//   for (unsigned int j = 0; j < Ain->size2; ++j) {
//     gsl::vector<T> v = gsl::matrix_column(Aout, j);
//     gsl::blas_scal(gsl::vector_get(e, j), &v);
//   }
// }

template <typename T>
void Equilibrate(gsl::matrix<T> *A, gsl::vector<T> *d, gsl::vector<T> *e,
                 bool compute_scaling) {
  T *dpr = d->data, *epr = e->data;
  if (compute_scaling) {
    if (A->size1 < A->size2) {
      gsl::vector_set_all(e, static_cast<T>(1));
      gsl::vector_set_all(d, static_cast<T>(0));
#pragma omp parallel for
      for (unsigned int i = 0; i < A->size1; ++i)
        for (unsigned int j = 0; j < A->size2; ++j)
          dpr[i] += std::fabs(gsl::matrix_get(A, i, j));
//           dpr[i] += gsl::matrix_get(A, i, j) * gsl::matrix_get(A, i, j);
      for (unsigned int i = 0; i < A->size1; ++i)
        dpr[i] = 1 / dpr[i];
//         dpr[i] = 1 / std::sqrt(dpr[i]);
    } else {
      gsl::vector_set_all(e, static_cast<T>(0));
      gsl::vector_set_all(d, static_cast<T>(1));
      for (unsigned int i = 0; i < A->size1; ++i)
        for (unsigned int j = 0; j < A->size2; ++j)
          epr[j] += std::fabs(gsl::matrix_get(A, i, j));
//           epr[j] += gsl::matrix_get(A, i, j) * gsl::matrix_get(A, i, j);
      for (unsigned int j = 0; j < A->size2; ++j)
        epr[j] = 1 / epr[j];
//         epr[j] = 1 / std::sqrt(epr[j]);
    }
  }
#pragma omp parallel for
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j) {
      gsl::matrix_set(A, i, j, gsl::matrix_get(A, i, j) * epr[j] * dpr[i]);
    }
  }
}

// template <typename T>
// void self_blas_gemv(CBLAS_TRANSPOSE_t Trans, T alpha, gsl::matrix<T> *A,
//                     gsl::vector<T> *x, T beta, gsl::vector<T> *y) {
//   if (Trans == CblasNoTrans) {
//     if (alpha == 1) {
// #pragma omp parallel for
//       for (unsigned int i = 0; i < A->size1; ++i) {
//         gsl::vector_set(y, i, gsl::vector_get(y, i) * beta);
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           gsl::vector_set(y, i, gsl::matrix_get(A, i, j) * gsl::vector_get(x, j) + gsl::vector_get(y, i));
//         }
//       }
//     } else {
// #pragma omp parallel for
//       for (unsigned int i = 0; i < A->size1; ++i) {
//         gsl::vector_set(y, i, gsl::vector_get(y, i) * beta);
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           gsl::vector_set(y, i, alpha * gsl::matrix_get(A, i, j) * gsl::vector_get(x, j) + gsl::vector_get(y, i));
//         }
//       }
//     }
//   } else {
//     if (alpha == 1) {
//       for (unsigned int j = 0; j < A->size2; ++j)
//         gsl::vector_set(y, j, gsl::vector_get(y, j) * beta);
//       for (unsigned int i = 0; i < A->size1; ++i) {
// //#pragma omp parallel for
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           gsl::vector_set(y, j, gsl::matrix_get(A, i, j) * gsl::vector_get(x, i) + gsl::vector_get(y, j));
//         }
//       }
//     } else {
//       for (unsigned int j = 0; j < A->size2; ++j)
//         gsl::vector_set(y, j, gsl::vector_get(y, j) * beta);
//       for (unsigned int i = 0; i < A->size1; ++i) {
// //#pragma omp parallel for
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           gsl::vector_set(y, j, alpha * gsl::matrix_get(A, i, j) * gsl::vector_get(x, i) + gsl::vector_get(y, j));
//         }
//       }
//     }
//   }
// }
// 
// template <typename T>
// void self_blas_gemm(CBLAS_TRANSPOSE_t TransB, T alpha,
//                     gsl::matrix<T> *A, gsl::matrix<T> *B, T beta , gsl::matrix<T> *C) {
//   if (TransB == CblasNoTrans) {
//     if (alpha == 1) {
//       for (unsigned int i = 0; i < C->size1; ++i) {
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           gsl::matrix_set(C, i, j, gsl::matrix_get(C, i, j) * beta);
//         }
//       }
//       for (unsigned int i = 0; i < A->size2; ++i) {
// //#pragma omp parallel for
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           for (unsigned int k = 0; k < C->size1; ++k) {
//             gsl::matrix_set(C, k, j, gsl::matrix_get(A, k, i) * gsl::matrix_get(B, i, j) + gsl::matrix_get(C, k, j));
//           }
//         }
//       }
//     } else {
//      for (unsigned int i = 0; i < C->size1; ++i) {
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           gsl::matrix_set(C, i, j, gsl::matrix_get(C, i, j) * beta);
//         }
//       }
//       for (unsigned int i = 0; i < A->size2; ++i) {
// //#pragma omp parallel for
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           for (unsigned int k = 0; k < C->size1; ++k) {
//             gsl::matrix_set(C, k, j, alpha * gsl::matrix_get(A, k, i) * gsl::matrix_get(B, i, j) + gsl::matrix_get(C, k, j));
//           }
//         }
//       }
//     }
//   } else {
//     if (alpha == 1) {
//       for (unsigned int i = 0; i < C->size1; ++i) {
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           gsl::matrix_set(C, i, j, gsl::matrix_get(C, i, j) * beta);
//         }
//       }
// #pragma omp parallel for
//       for (unsigned int i = 0; i < C->size2; ++i) {
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           for (unsigned int k = 0; k < C->size1; ++k) {
//             gsl::matrix_set(C, k, i, gsl::matrix_get(A, k, j) * gsl::matrix_get(B, i, j) + gsl::matrix_get(C, k, i));
//           }
//         }
//       }
//     } else {
//       for (unsigned int i = 0; i < C->size1; ++i) {
//         for (unsigned int j = 0; j < C->size2; ++j) {
//           gsl::matrix_set(C, i, j, gsl::matrix_get(C, i, j) * beta);
//         }
//       }
// #pragma omp parallel for
//       for (unsigned int i = 0; i < C->size2; ++i) {
//         for (unsigned int j = 0; j < A->size2; ++j) {
//           for (unsigned int k = 0; k < C->size1; ++k) {
//             gsl::matrix_set(C, k, i, alpha * gsl::matrix_get(A, k, j) * gsl::matrix_get(B, i, j) + gsl::matrix_get(C, k, i));
//           }
//         }
//       }
//     }
//   }
// }


#endif  // SINKHORN_KNOPP_H_

