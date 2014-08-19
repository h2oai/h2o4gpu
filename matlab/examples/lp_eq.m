function results = lp_eq(m, n, rho, quiet)
%%LP_EQ Test POGS on an equality constrained LP.
%   Compares POGS to CVX when solving the problem
%
%     minimize    c^T * x
%     subject to  Ax = b
%                 x >= 0.
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [A; c^T] * x,
%
%   where g(x_i)        = I(x_i >= 0)
%         f_{1..m}(y_i) = I(y_i = b_i)
%         f_{m+1}(y_i)  = y_i.
%
%   Test data are generated as follows
%     - The entries in A and c are drawn uniformly in [0, 1].
%     - To generate b, we first choose a vector v with entries drawn
%       uniformly from [0, 1], we assign b = A * v. This ensures that b is
%       in the range of A.
%
%   results = lp_eq()
%   results = lp_eq(m, n, rho, quiet)
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
elseif m > n
  error('A must be a fat matrix')
end
if nargin < 3
  rho = 1;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

A = 1 / n * randn(m, n);
b = A * rand(n, 1);
c = rand(n, 1);

% Declare proximal operators.
prox_g = @(x, rho) max(x, 0);
prox_f = @(x, rho) [b; x(end) - 1 / rho(end)];
obj_fn = @(x, y) y(end);

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 10000;
params.RELTOL = 1e-3;

% Solve using POGS.
tic
[x_pogs, ~, ~, n_iter] = pogs(prox_f, prox_g, obj_fn, [A; c'], params);
time_pogs = toc;
c' * x_pogs

% Solve using CVX.
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(c' * x_cvx);
  subject to
    A * x_cvx == b;
    x_cvx >= 0;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, [A; c'] * x_pogs) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_pogs - x_cvx) / norm(x_cvx);
results.max_violation = max([abs(b - A * x_pogs); max(-x_pogs, 0)]);
results.avg_violation = mean([abs(b - A * x_pogs); max(-x_pogs, 0)]);
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
