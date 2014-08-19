function results = entropy(m, n, rho, quiet)
%%ENROPY Test POGS on an entropy maximization problem.
%   Compares POGS to CVX when solving the problem
%
%     minimize    sum_{i=1}^n x_i * log(x_i)
%     subject to  sum(x) = 1
%                 Ax <= b
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = [A; 1] * x,
%
%   where g(x_i)        = x_i * log(x_i)
%         f_{1..m}(y_i) = I(y_i <= b_i)
%         f_{m+1}(y_i)  = I(y_i = 1)
%
%   Test data are generated as follows
%     - The entries in A are generated from a uniform [0, 1] distribution.
%     - To generate b, we first choose a vector v with entries drawn
%       uniformly from [0, 1]. b is then generated as b = F * v + w, where
%       w is drawn uniformly from the range [0, 1].
%
%   results = entropy()
%   results = entropy(m, n, rho, quiet)
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
  m = 20;
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

F = rand(m, n) / n;
b = F * rand(n, 1) + rand(m, 1);
A = [F; ones(1, n)];

% Declare proximal operators.
prox_g = @(x, rho) 1 ./ rho .* lambertw(exp(x .* rho - 1) .* rho);
prox_f = @(y, rho) [min(y(1:end-1), b); 1];
obj_fn = @(x, y) x' * log(x);

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
    variable x_cvx(n)
    minimize(-sum(entr(x_cvx)))
    F * x_cvx <= b
    sum(x_cvx) == 1
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_pogs, A * x_pogs) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_pogs - x_cvx) / norm(x_cvx);
results.max_violation = max(abs(sum(x_pogs) - 1), ...
                            max(max(F * x_pogs - b, 0)));
results.avg_violation = mean(max(F * x_pogs - b, 0));
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
