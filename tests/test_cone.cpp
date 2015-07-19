#include "pogs.h"
#include "util.h"
#include "test_util.h"

#define VERBOSE 0
#define ABS_TOL 1e-5
#define REL_TOL 1e-5
#define MAX_ITER 1000

#define TEST_EPS 1e-4

/*
 *  min. [1 0] [x_1 x_2]'
 *  s.t.
 *       [-1 -1]               [0]
 *       [ 1 -1] [x_1 x_2]' <= [0]
 *       [ 0  1]               [2]
 */

#define LP_OPTVAL -2.000000000

template <typename T>
void lp_cone_row_direct() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., -1., 1., -1., 0., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  // Dense
  {
    pogs::MatrixDense<T> A_('r', m, n, A.data());
    pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
  }
}

template <typename T>
void lp_cone_row_indirect() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., -1., 1., -1., 0., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  // Dense
  {
    pogs::MatrixDense<T> A_('r', m, n, A.data());
    pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
  }

  // Sparse
  {
    std::vector<int> row_ptr, col_ind;
    std::vector<T> val;

    row2csr(A, m, n, &val, &row_ptr, &col_ind);

    pogs::MatrixSparse<T> A_('r', static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(val.size()), val.data(), row_ptr.data(),
        col_ind.data());
    pogs::PogsIndirectCone<T, pogs::MatrixSparse<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
  }
}

template <typename T>
void lp_cone_col_direct() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., 1., 0., -1., -1., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('c', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(MAX_ITER);
  pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
  pogs_data.SetRelTol(static_cast<T>(REL_TOL));
  pogs_data.SetVerbose(VERBOSE);
  pogs::PogsStatus status = pogs_data.Solve(b, c);

  TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
  TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
}

template <typename T>
void lp_cone_col_indirect() {
  size_t m = 3, n = 2;

  std::vector<T> A = {-1., 1., 0., -1., -1., 1};
  std::vector<T> b = {0., 0., 2.};
  std::vector<T> c = {1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeNonNeg, {0, 1, 2}}};
  std::vector<ConeConstraint> Kx;

  // Dense
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsIndirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
  }

  // Sparse
  {
    std::vector<int> col_ptr, row_ind;
    std::vector<T> val;

    col2csc(A, m, n, &val, &col_ptr, &row_ind);

    pogs::MatrixSparse<T> A_('c', static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(val.size()), val.data(), col_ptr.data(),
        row_ind.data());
    pogs::PogsIndirectCone<T, pogs::MatrixSparse<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), LP_OPTVAL, TEST_EPS);
  }
}


/*
 *  min. [0 1 0] [x_1 x_2 x_3]'
 *  s.t.
 *      [4]   [ 1  0  0]
 *      [0]   [-1  0  0]
 *      [0] - [ 0 -1 -1] [x_1 x_2 x_3]' \in {0, SOC(3)}
 *      [0]   [ 0 0   1]
 */

#define SOC_OPTVAL -5.65685425
template <typename T>
void soc_cone_row_direct() {
  size_t m = 4, n = 3;

  std::vector<T> A = { 1.,  0.,  0.,
                      -1.,  0.,  0.,
                       0., -1., -1.,
                       0.,  0.,  1.};
  std::vector<T> b = {4., 0., 0., 0.};
  std::vector<T> c = {0., 1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeZero, {0}}, {kConeSoc, {1, 2, 3}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('r', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(MAX_ITER);
  pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
  pogs_data.SetRelTol(static_cast<T>(REL_TOL));
  pogs_data.SetVerbose(VERBOSE);
  pogs::PogsStatus status = pogs_data.Solve(b, c);

  TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
  TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
}

template <typename T>
void soc_cone_row_indirect() {
  size_t m = 4, n = 3;

  std::vector<T> A = { 1.,  0.,  0.,
                      -1.,  0.,  0.,
                       0., -1., -1.,
                       0.,  0.,  1.};
  std::vector<T> b = {4., 0., 0., 0.};
  std::vector<T> c = {0., 1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeZero, {0}}, {kConeSoc, {1, 2, 3}}};
  std::vector<ConeConstraint> Kx;

  // Dense
  {
    pogs::MatrixDense<T> A_('r', m, n, A.data());
    pogs::PogsIndirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
  }

  // Sparse
  {
    std::vector<int> row_ptr, col_ind;
    std::vector<T> val;

    row2csr(A, m, n, &val, &row_ptr, &col_ind);
    pogs::MatrixSparse<T> A_('r', static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(val.size()), val.data(), row_ptr.data(),
        col_ind.data());
    pogs::PogsIndirectCone<T, pogs::MatrixSparse<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
  }

}

template <typename T>
void soc_cone_col_direct() {
  size_t m = 4, n = 3;

  std::vector<T> A = { 1., -1.,  0.,  0.,
                       0.,  0., -1.,  0.,
                       0.,  0., -1.,  1.};
  std::vector<T> b = {4., 0., 0., 0.};
  std::vector<T> c = {0., 1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeZero, {0}}, {kConeSoc, {1, 2, 3}}};
  std::vector<ConeConstraint> Kx;

  pogs::MatrixDense<T> A_('c', m, n, A.data());
  pogs::PogsDirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

  pogs_data.SetMaxIter(MAX_ITER);
  pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
  pogs_data.SetRelTol(static_cast<T>(REL_TOL));
  pogs_data.SetVerbose(VERBOSE);
  pogs::PogsStatus status = pogs_data.Solve(b, c);

  TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
  TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
}

template <typename T>
void soc_cone_col_indirect() {
  size_t m = 4, n = 3;

  std::vector<T> A = { 1., -1.,  0.,  0.,
                       0.,  0., -1.,  0.,
                       0.,  0., -1.,  1.};
  std::vector<T> b = {4., 0., 0., 0.};
  std::vector<T> c = {0., 1., 0.};

  std::vector<ConeConstraint> Ky = {{kConeZero, {0}}, {kConeSoc, {1, 2, 3}}};
  std::vector<ConeConstraint> Kx;

  // Dense
  {
    pogs::MatrixDense<T> A_('c', m, n, A.data());
    pogs::PogsIndirectCone<T, pogs::MatrixDense<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
  }

  // Sparse
  {
    std::vector<int> col_ptr, row_ind;
    std::vector<T> val;

    col2csc(A, m, n, &val, &col_ptr, &row_ind);

    pogs::MatrixSparse<T> A_('c', static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(val.size()), val.data(), col_ptr.data(),
        row_ind.data());
    pogs::PogsIndirectCone<T, pogs::MatrixSparse<T> > pogs_data(A_, Kx, Ky);

    pogs_data.SetMaxIter(MAX_ITER);
    pogs_data.SetAbsTol(static_cast<T>(ABS_TOL));
    pogs_data.SetRelTol(static_cast<T>(REL_TOL));
    pogs_data.SetVerbose(VERBOSE);
    pogs::PogsStatus status = pogs_data.Solve(b, c);

    TEST_EXPECT_EQ(status, pogs::POGS_SUCCESS);
    TEST_EXPECT_EQ_EPS(pogs_data.GetOptval(), SOC_OPTVAL, TEST_EPS);
  }
}


int main() {
  // LP
  lp_cone_row_direct<double>();
  lp_cone_row_indirect<double>();
  lp_cone_col_direct<double>();
  lp_cone_col_indirect<double>();

  lp_cone_row_direct<float>();
  lp_cone_row_indirect<float>();
  lp_cone_col_direct<float>();
  lp_cone_col_indirect<float>();

  // SOC
  soc_cone_row_direct<double>();
  soc_cone_row_indirect<double>();
  soc_cone_col_direct<double>();
  soc_cone_col_indirect<double>();

  soc_cone_row_direct<float>();
  soc_cone_row_indirect<float>();
  soc_cone_col_direct<float>();
  soc_cone_col_indirect<float>();
}

