function results = lasso(m, n, rho, quiet)
%%LASSO Test POGS on Lasso.
%   Compares POGS to CVX when solving the problem
%
%     minimize    (1/2) * ||Ax - b||_2^2 + \lambda * ||x||_1
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x,
%
%   where g(x_i) = lambda * |x_i|
%         f(y_i) = (1/2) * (y_i - b_i) ^ 2
%
%   Test data are generated as follows
%     - The entries in A are generated from a normal N(0, 1) distribution.
%     - To generate b, we first choose a vector v with entries
%       that are zero with probability 0.8 and otherwise drawn from
%       a normal N(0, 1) distribution. b is then generated as
%       b = A * v + w, where w is Gaussian noise with mean zero and 
%       variance 0.5.
%
%   results = lasso()
%   results = lasso(m, n, rho, quiet)
% 
%   Optional Inputs: (m, n), rho, quiet
%
%   Optional Inputs:
%   (m, n)    - (default 2000, 200) Dimensions of the matrix A.
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

A = randn(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.1 * randn(m, 1);
lambda = 1 / 3 * max(abs(A' * b));

% Declare proximal operators.
prox_g = @(x, rho) max(x - lambda ./ rho, 0) - max(-x - lambda ./ rho, 0);
prox_f = @(x, rho) (rho .* x + b) ./ (1 + rho);
obj_fn = @(x, y) 1 / 2 * norm(y - b) ^ 2 + lambda * norm(x, 1);

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
params.RELTOL = 1e-3;
params.ABSTOL = 1e-4;
params.indirect = true;

% Solve using POGS.
tic
[x_pogs, y, ~, n_iter] = pogs(prox_f, prox_g, obj_fn, (A), params);
time_pogs = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variables x_cvx(n) y(m)
  minimize(1 / 2 * (y - b)' * (y - b) + lambda * norm(x_cvx, 1));
  subject to
    A * x_cvx == y;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_pogs - x_cvx) / norm(x_cvx);
results.max_violation = nan;
results.avg_violation = nan;
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
