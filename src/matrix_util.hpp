#ifndef MATRIX_UTIL_HPP_
#define MATRIX_UTIL_HPP_

#include <algorithm>
#include <cstdlib>

// Block row to column major.
template <typename T>
void RowToColMajor(const T *A_in, size_t m, size_t n, T *A_out) {
  const size_t kBlkSize = 64;

  size_t brow = (m + kBlkSize - 1) / kBlkSize;
  size_t bcol = (n + kBlkSize - 1) / kBlkSize;

  for (size_t i_blk = 0; i_blk < brow; ++i_blk) {
    for (size_t j_blk = 0; j_blk < bcol; ++j_blk) {
      for (size_t i = 0; i < std::min(kBlkSize, m - i_blk * kBlkSize); ++i) {
        for (size_t j = 0; j < std::min(kBlkSize, n - j_blk * kBlkSize); ++j) {
          size_t i_global = i_blk * kBlkSize + i;
          size_t j_global = j_blk * kBlkSize + j;
          A_out[i_global + j_global * m] = A_in[i_global * n + j_global];
        }
      }
    }
  }
}

// Block column to row major.
template <typename T>
void ColToRowMajor(const T *A_in, size_t m, size_t n, T *A_out) {
  RowToColMajor(A_in, n, m, A_out);
}

#endif  // MATRIX_UTIL_HPP_

