function results = lp_ineq(m, n, rho, quiet)
%%LP_INEQ Test H2OGPUML on an inequality constrained LP.
%   Compares H2OGPUML to CVX when solving the problem
%
%     minimize    c^T * x
%     subject to  Ax <= b.
%
%   We transform this problem to
%
%     minimize    f(y) + g(x)
%     subject to  y = A * x,
%
%   where g_i(x_i) = c_i * x_i
%         f_i(y_i) = I(y_i <= b_i).
%
%   Test data are generated as follows
%     - The entries in c are chosen uniformly in the interval [0, 1].
%     - The first m-n entries in A are generated uniformly in
%       [-1/n, 0], this ensures that the problem will be feasible (since 
%       x -> \infty will always be a solution). In addition the last m
%       entries of A are set to the negative of the identity matrix. Since 
%       the the vector c is non-negative, this added constraint ensures 
%       that the problem is bounded. 
%     - To generate b, we first choose a vector v with entries drawn
%       uniformly from [0, 1], we assign b = A * v and add Gaussian
%       noise. The vector b is chosen this way, so that the solution
%       x^\star has reasonably uniform entries.
%
%   results = lp_ineq()
%   results = lp_ineq(m, n, rho, quiet)
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
%                   (h2ogpuml_optval - cvx_optval) / abs(cvx_optval).
%                 + rel_err_soln: Relative difference in solution between
%                   CVX and H2OGPUML, defined as 
%                   norm(x_h2ogpuml - x_cvx) / norm(x_cvx).
%                 + max_violation: Maximum constraint violation (nan if 
%                   problem has no constraints).
%                 + avg_violation: Average constraint violation.
%                 + time_h2ogpuml: Time required by H2OGPUML to solve problem.
%                 + time_cvx: Time required by CVX to solve problem.
%

% Parse inputs.
if nargin < 2
  m = 2000;
  n = 200;
elseif m < n
  error('A must be a skinny matrix')
end
if nargin < 3
  rho = 1;
end
if nargin < 4
  quiet = false;
end

% Initialize Data.
rng(0, 'twister')

A = -[4 / n * rand(m - n, n); eye(n)];
b = A * rand(n, 1) + 0.2 * rand(m, 1);
c = rand(n, 1);

% Declare proximal operators.
prox_g = @(x, rho) x - c ./ rho;
prox_f = @(x, rho) min(b, x);
obj_fn = @(x, y) c' * x;

% Initialize H2OGPUML input.
params.rho = rho;
params.quiet = quiet;
params.MAXITR = 1000;

% Solve using H2OGPUML.
tic
x_h2ogpuml = h2ogpuml(prox_f, prox_g, obj_fn, A, params);
time_h2ogpuml = toc;

% Solve using CVX.
tic
cvx_begin quiet
  variable x_cvx(n)
  minimize(c' * x_cvx);
  subject to
    A * x_cvx <= b;
cvx_end
time_cvx = toc;

% Compute error metrics.
results.rel_err_obj = ...
    (obj_fn(x_h2ogpuml, A * x_h2ogpuml) - cvx_optval) / abs(cvx_optval);
results.rel_diff_soln = norm(x_h2ogpuml - x_cvx) / norm(x_cvx);
results.max_violation = abs(min(min(b - A * x_h2ogpuml), 0)) / norm(x_h2ogpuml);
results.avg_violation = mean(abs(min(b - A * x_h2ogpuml, 0)));
results.time_h2ogpuml = time_h2ogpuml;
results.time_cvx = time_cvx;

% Print error metrics
if ~quiet
  fprintf('\nRelative Error of Objective: %e\n', results.rel_err_obj)
  fprintf('Relative Difference in Solution: %e\n', results.rel_diff_soln)
  fprintf('Maximum Constraint Violation: %e\n', results.max_violation)
  fprintf('Average Constraint Violation: %e\n', results.avg_violation)
  fprintf('Time H2OGPUML: %e\n', results.time_h2ogpuml)
  fprintf('Time CVX: %e\n', results.time_cvx)
end

end
