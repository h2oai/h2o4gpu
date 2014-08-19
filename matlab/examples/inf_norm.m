function results = inf_norm(m, n, rho, quiet)
%%INF_NORM Test POGS on minimizing the L-inf norm.
%   Compares POGS to CVX when solving the problem
%
%     minimize    ||Ax - b||_inf
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [A -1;-A -1] * x,
%
%   where g_{1..n}(x_i)    = 0
%         g_{n+1}(x_i)     = x_i
%         f_{1..m}(y_i)    = I(y_i <= b_i)
%         f_{m+1..2m}(y_i) = I(y_i <= -b_i)
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
%   results = inf_norm()
%   results = inf_norm(m, n, rho, quiet)
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
  rho = 1;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

A = 1 / n * rand(m, n);
b = A * rand(n, 1) + 2 * randn(m, 1);
B = [A -ones(m, 1); -A -ones(m, 1)];

% Declare proximal operators.
prox_g = @(x, rho) [x(1:end-1); x(end) - 1 / rho(end)];
prox_f = @(y, rho) [min(y(1:m), b); min(y(m+1:end), -b)];
obj_fn = @(x, y) x(end);

% Initialize POGS input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
params.RELTOL = 1e-3;
params.ABSTOL = 1e-4;
params.norml = true;

% Solve using POGS.
tic
[x_pogs, ~, ~, n_iter] = pogs(prox_f, prox_g, obj_fn, B, params);
time_pogs = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variables x_cvx(n) t
  minimize(t);
    -t <= A * x_cvx - b <= t
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, B * x_pogs) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_pogs(1:end - 1) - x_cvx) / norm(x_cvx);
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
