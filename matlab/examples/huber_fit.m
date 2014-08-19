function results = huber_fit(m, n, rho, quiet)
%%HUBER_FIT Test POGS on Huber fitting.
%   Compares POGS to CVX when solving the problem
%
%     minimize    huber(Ax - b)
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x,
%
%   where g(x_i) = 0
%         f(y_i) = huber(y_i - b_i)
%
%   Test data are generated as follows
%     - The entries in A are generated from a normal N(0, 1) distribution.
%     - The vector b is generated from a normal N(0, 1) distribution with
%       0.95 probability and drawn uniformaly from the interval [0, 10]
%       with probability 0.05.
%
%   results = huber_fit()
%   results = huber_fit(m, n, rho, quiet)
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
  m = 2000;
  n = 200;
end
if nargin < 3
  rho = 1;1e1;1e-1;5e1;1e1;1e2;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

A = rand(m, n);
p = rand(m, 1);
b = randn(m, 1) .* (p <= 0.95) + 10 * rand(m, 1) .* (p > 0.95);

% Declare proximal operators.
prox_g = @(x, rho) x;
prox_f = @(y, rho) (abs(y - b) < 1 + 1 ./ rho) .* (y - b) .* rho ./ (1 + rho) + ...
                   (abs(y - b) >= 1 + 1 ./ rho) .* (y - b - sign(y - b) ./ rho) + b;
obj_fn = @(x, y) sum((abs(y - b) < 1) .* ((y - b) .^ 2) / 2 + (abs(y - b) >= 1) .* (abs(y - b) - 0.5));

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
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
  minimize(1/2*sum(huber(A * x_cvx - b)));
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
