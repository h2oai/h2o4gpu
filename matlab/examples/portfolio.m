function results = portfolio(m, n, rho, quiet)
%%PORTFOLIO Test H2OAIGLM on a portfolio problem.
%   Compares H2OAIGLM to CVX when solving the problem
%
%     minimize    r^T x + (\gamma/2) x^T (FF^T + D) x
%     subject to  x >= 0, 1^T x = 1
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [F 1]' * x,
%
%   where g(x)          = I(x >= 0) + r' * x + (gamma / 2) x' * D * x
%         f_{1..n}(y_i) = (gamma / 2) * y_i ^ 2
%         f_{m+1}(y_i)  = I(y_i = 1)
%
%   Test data are generated as follows
%     - The entries in F are generated from a normal N(0, 1/m^2)
%       distribution.
%     - The diagonal entries in D are generated uniformly from the interval
%       [0, 1 / n]. Off diagonal entries are equal to zero.
%     - The vector r is generated from a normal N(0, 1/n^2) distribution.
%
%   results = portfolio()
%   results = portfolio(m, n, rho, quiet)
% 
%   Optional Inputs: (m, n), rho, quiet
%
%   Optional Inputs:
%   (m, n)    - (default 2000, 200) Dimensions of the matrix F.
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
  m = 200;
  n = 2000;
end
if nargin < 3
  rho = 1;7e1;0.540806;0.5534;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

gamma = 1;

F = randn(n, m) / n;
A = [F ones(n, 1)]';
d = rand(n, 1);
r = -rand(n, 1);

% Declare proximal operators.
prox_g = @(x, rho) max(0, (x .* rho - r) ./ (d * gamma + rho));
prox_f = @(x, rho) [x(1:end-1) .* rho(1:end-1) ./ (rho(1:end-1) + gamma); 1];
obj_fn = @(x, y) r' * x + gamma / 2 * (sum(d .* x .^ 2) + norm(y(1:end-1))^2);

% Initialize H2OAIGLM input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;
params.RELTOL = 1e-3;
params.ABSTOL = 1e-4;

% Solve using H2OAIGLM.
tic
[x_h2oaiglm, ~, ~, n_iter] = h2oaiglm(prox_f, prox_g, obj_fn, A, params);
time_h2oaiglm = toc;
sum(x_h2oaiglm)

% Solve using CVX.
tic
cvx_begin quiet
  variables x_cvx(n)
  minimize(r' * x_cvx + gamma / 2 * (sum_square_abs(F' * x_cvx) + ...
           sum(d .* x_cvx .^ 2)));
  subject to
    x_cvx >= 0;
    sum(x_cvx) == 1;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_h2oaiglm, A * x_h2oaiglm) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_h2oaiglm - x_cvx) / norm(x_cvx);
results.max_violation = max(abs(sum(x_h2oaiglm) - 1), max(max(-x_h2oaiglm, 0))) / ...
    norm(x_h2oaiglm);
results.avg_violation = mean([abs(sum(x_h2oaiglm) - 1); max(-x_h2oaiglm, 0)]) / ...
    norm(x_h2oaiglm);
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
