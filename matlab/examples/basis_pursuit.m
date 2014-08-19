function results = basis_pursuit(m, n, rho, quiet)
%%BASIS_PURSIUT Test POGS on Basis Pursuit.
%   Compares POGS to CVX when solving the problem
%
%     minimize    ||x||_1
%     subject to  Ax = b
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x,
%
%   where g(x_i) = |x_i|
%         f(y_i) = I(y_i = b_i)
%
%   Test data are generated as follows
%     - The entries in A are generated from a normal N(0, 1 / n^2) 
%       distribution.
%     - To generate b, we first choose a vector v with entries
%       that are zero with probability 0.8 and otherwise drawn from
%       a normal N(0, 1) distribution. b is then generated as
%       b = A * v + w, where w is Gaussian noise with mean zero and 
%       variance 0.5.
%
%   results = basis_pursuit()
%   results = basis_pursuit(m, n, rho, quiet)
% 
%   Optional Inputs: (m, n), rho, quiet
%
%   Optional Inputs:
%   (m, n)    - (default 200, 2000) Dimensions of the matrix A.
%   
%   rho       - (default 1.0) Penalty parameter to proximal operators.
% 
%   quiet     - (default false) Set flag to true, to disable output to
%               console.
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
  m = 200;
  n = 2000;
end
if nargin < 3
  rho = 1;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

A = 1 / n * randn(m, n) .* (rand(m, n) > 0.8);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1));

% Declare proximal operators.
prox_g = @(x, rho) max(x - 1 ./ rho, 0) + min(x + 1 ./ rho, 0);
prox_f = @(x, rho) b;
obj_fn = @(x, y) norm(x, 1);

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 10000;
params.RELTOL = 1e-3;
params.ABSTOL = 1e-4;

% Solve using POGS.
tic
[x_pogs, ~, ~, n_iter] = pogs(prox_f, prox_g, obj_fn, A, params);
time_pogs = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variables x_cvx(n)
  minimize(norm(x_cvx, 1));
  subject to
    A * x_cvx == b;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_pogs - x_cvx) / norm(x_cvx);
results.max_violation = norm(A * x_pogs - b, 1) / norm(x_pogs);
results.avg_violation = mean(abs(A * x_pogs - b)) / norm(x_pogs);
results.time_pogs = time_pogs;
results.time_cvx = time_cvx;
results.n_iter = n_iter;

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
