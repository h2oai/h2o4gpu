function results = pwl(m, n, rho, quiet)
%%PWL Test H2OAIGLM on Piecewise-Linear fitting.
%   Compares H2OAIGLM to CVX when solving the problem
%
%     minimize    max(Ax - b)
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [A -1] * x,
%
%   where g_{1..n}(x_i)  = 0
%         g_{n+1}(x_i)   = x_i
%         f(y_i)         = I(y_i <= b_i)
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
%   results = pwl()
%   results = pwl(m, n, rho, quiet)
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
%                   (h2oaiglm_optval - cvx_optval) / abs(cvx_optval).
%                 + rel_err_soln: Relative difference in solution between
%                   CVX and H2OAIGLM, defined as 
%                   norm(x_h2oaiglm - x_cvx) / norm(x_cvx).
%                 + max_violation: Maximum constraint violation (nan if 
%                   problem has no constraints).
%                 + avg_violation: Average constraint violation.
%                 + time_h2oaiglm: Time required by H2OAIGLM to solve problem.
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

A = [1 / n * rand(m - n, n); -eye(n)];
b = A * rand(n, 1) + 2 * randn(m, 1);
B = [A -ones(m, 1)];

% Declare proximal operators.
prox_g = @(x, rho) [x(1:end-1); x(end) - 1 / rho(end)];
prox_f = @(x, rho) min(x, b);
obj_fn = @(x, y) x(end);

% Initialize H2OAIGLM input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 10000;
params.RELTOL = 1e-3;
params.ABSTOL = 1e-4;
params.norml = true;

% Solve using H2OAIGLM.
tic
[x_h2oaiglm, ~, ~, n_iter] = h2oaiglm(prox_f, prox_g, obj_fn, B, params);
time_h2oaiglm = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variables x_cvx(n) t
  minimize(t);
    A * x_cvx - b <= t
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (max(A * x_h2oaiglm(1:end-1) - b) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_h2oaiglm(1:end - 1) - x_cvx) / norm(x_cvx);
results.max_violation = nan;
results.avg_violation = nan;
results.time_h2oaiglm = time_h2oaiglm;
results.time_cvx = time_cvx;
results.n_iter = n_iter;

% Print error metrics.
if ~quiet
  fprintf('\nRelative Error of Objective: %e\n', results.rel_err_obj)
  fprintf('Relative Difference in Solution: %e\n', results.rel_diff_soln)
  fprintf('Maximum Constraint Violation: %e\n', results.max_violation)
  fprintf('Average Constraint Violation: %e\n', results.avg_violation)
  fprintf('Time H2OAIGLM: %e\n', results.time_h2oaiglm)
  fprintf('Time CVX: %e\n', results.time_cvx)
end
end
