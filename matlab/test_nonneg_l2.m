function results = test_nonneg_l2(m, n, rho, quiet, save_mat)
%%TEST_NONNEG_L2 Test POGS on non-negative least squares.
%   Compares POGS to CVX when solving the problem
%
%     minimize    (1/2) ||Ax - b||_2^2
%     subject to  x >= 0.
%
%   We transform this problem into
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x,
%
%   where g_i(x_i) = I(x_i >= 0)
%         f_i(y_i) = (1/2) * (y_i - b_i) ^ 2.
%
%   Test data are generated as follows
%     - Entries in A are generated uniformly at random in [0, 1/n].
%     - Entries in b are generated such that the optimal unconstrained
%       solution x^\star is approximately equal to [1..1 -1..-1]^T, 
%       guaranteeing that some constraints will be active.
%
%   results = test_nonneg_l2()
%   results = test_nonneg_l2(m, n, rho, quiet, save_mat)
% 
%   Optional Inputs: (m, n), rho, quiet, save_mat
%
%   Optional Inputs:
%   (m, n)    - (default 2000, 200) Dimensions of the matrix A.
%   
%   rho       - (default 1.0) Penalty parameter to proximal operators.
% 
%   quiet     - (default false) Set flag to true, to disable output to
%               console.
%
%   save_mat  - (default false) Save data matrices to MatrixMarket files.
%
%   Outputs:
%   results   - Structure containg test results. Fields are:
%                 + rel_err_obj: Relative error of the objective, as
%                   compared to the solution obtained from CVX, defined as
%                   (pogs_optval - cvx_optval) / abs(cvx_optval).
%                 + rel_err_soln: Relative difference in solution between
%                   CVX and POGS, defined as 
%                   norm(x_pogs - x_cvx) / norm(x_cvx).
%                 + max_violation: Maximum constraint violation (nan if
%                   problem has no constraints).
%                 + avg_violation: Average constraint violation.
%                 + time_pogs: Time required by POGS to solve problem.
%                 + time_cvx: Time required by CVX to solve problem.
%

% Parse inputs.
if nargin < 2
  m = 2000;
  n = 200;
end
if nargin < 3
  rho = 1.0;
end
if nargin < 4
  quiet = false;
end
if nargin < 5
  save_mat = false;
end

% Initialize Variables.
rng(0, 'twister')

n_half = floor(2/3 *  n);
A = 2 / n * rand(m, n);
b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.01 * randn(m, 1);

% Export Matrices
if save_mat
  mmwrite('data/A_nonneg_l2.dat', A, 'Matrix A for test_nonneg_l2.m')
  mmwrite('data/b_nonneg_l2.dat', b, 'Matrix b for test_nonneg_l2.m')
end

% Declare proximal operators.
g_prox = @(x, rho) max(x, 0);
f_prox = @(x, rho) (x .* rho + b) ./ (1 + rho);
obj_fn = @(x, y) 1 / 2 * norm(A * x - b) ^ 2;

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
params.RELTOL = 1e-3;

% Solve using POGS.
tic
x_pogs = pogs(f_prox, g_prox, obj_fn, A, params);
time_pogs = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(1 / 2 * (A * x_cvx - b)' * (A * x_cvx - b));
  subject to
    x_cvx >= 0;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / cvx_optval;
results.rel_diff_soln = norm(x_pogs - x_cvx) / norm(x_cvx);
results.max_violation = abs(min(min(x_pogs), 0));
results.avg_violation = mean(abs(min(x_pogs, 0)));
results.time_pogs = time_pogs;
results.time_cvx = time_cvx;

% Print error metrics.
if ~quiet
  fprintf('\nRelative Error of Objective: %e\n', results.rel_err_obj)
  fprintf('Relative Difference in Solution: %e\n', results.rel_diff_soln)
  fprintf('Maximum Constraint Violation: %e\n', results.max_violation)
  fprintf('Average Constraint Violation: %e\n', results.avg_violation)
  fprintf('Time POGS: %e\n', results.time_pogs)
  fprintf('Time CVX: %e\n', results.time_cvx)
end

end
