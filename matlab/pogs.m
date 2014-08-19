function [x, y, factors, n_iter] = pogs(prox_f, prox_g, obj_fn, A, params, factors)
%%POGS Generic graph projection splitting solver.
%   Solves problems in the form
%
%     minimize    f(y) + g(x),
%     subject to  y = Ax.
% 
%   where the proximal operators of the functions f and g are known.
%
%   [x, factors] = pogs(prox_f, prox_g, obj_fn, A)
%   [x, factors] = pogs(prox_f, prox_g, obj_fn, A, params, factors)
% 
%   Optional Inputs: params, factors
%
%   Inputs:
%   prox_f    - Function handle or cell array of function handles, which
%               calculates returns the proximal operator of the function f.
%               If prox_f is a function handle, then it must accept vector
%               arguments and if prox_f is a cell array, then it must have
%               dimension equal to the number of rows in A. The function
%               handles must accept input in the form prox_f(x_i, rho), 
%               where rho is the penalty parameter.
%
%   prox_g    - Proximal operator for the function g. See description of
%               prox_f, with only difference that the cell array must have
%               dimension equal to the number of columns in A.
%
%   obj_fn    - Function handle which satsifies obj_fn(x, y) = f(y) + g(x).
%
%   A         - Complicating constraint matrix of dimension m * n.
%
%   Optional Inputs:
%   params    - Structure of parameters, containing any of the following
%               fields:
%                 + ABSTOL (default 1e-4): Absolute tolerance to which the
%                   problem should be solved.
%                 + RELTOL (default 1e-2): Relative tolerance to which the
%                   problem should be solved.
%                 + MAXITR (default 10000): Maximum number of iteratios
%                   that the solver should be run for.
%                 + rho (default 1.0): Penalty parameter for proximal
%                   operator.
%                 + quiet (default false): Set flag to true, to disable
%                   output to console.
%                 + adaptive_rho (default true): Adaptively choose rho.
%                 + indirect (default false): Uses LSQR instead of LDL (not implemented yet).
%                 + approx_res (default false): Use approximate residuals for stopping.
%
%   factors   - Structure containing pre-computed factors. If any
%               one field is missing, then all of them will be re-computed.
%               The fields are:
%                 + L, D, P: Cholesky decomposition factors of (I + A' * A) 
%                   or (I + A * A'), whichever results in a smaller matrix.
%                 + AA: The result of (A' * A) or (A * A'), whichever
%                   results in the smaller dimension.
%
%   Outputs:
%   x         - The partial solution x^\star to the optimization problem
%
%   y         - The partial solution y^\star to the optimization problem
%   
%   factors   - A structure of Cholesky decomposition factors.
%               See description of corresponding input. 
%
%   References: 
%   http://www.stanford.edu/~boyd/papers/block_splitting.html 
%     Block Splitting for Distributed Optimization -- N. Parikh and S. Boyd
%
%   http://www.stanford.edu/~boyd/papers/block_splitting.html
%     Distributed Optimization and Statistical Learning via the Alternating
%     Direction Method of Multipliers -- S. Boyd, N. Parikh, E. Chu,
%     B. Peleato, and J. Eckstein
%
%   http://www.stanford.edu/~boyd/papers/prox_algs.html
%     Proximal Algorithms -- N. Parikh and S. Boyd
%
%   Authors:
%   Neal Parikh, Chris Fougner.
%

% Parse Input.
if nargin < 4
  error('Not enough arguments.')
end
if nargin < 5
  params = [];
end
if nargin < 6
  factors = [];
end

ABSTOL = get_or_default(params, 'ABSTOL', 1e-4);
RELTOL = get_or_default(params, 'RELTOL', 1e-3);
MAXITR = get_or_default(params, 'MAXITR', 10000);
quiet = get_or_default(params, 'quiet', false);
norml = get_or_default(params, 'norml', true);
ada_rho = get_or_default(params, 'adaptive_rho', true);
approx_res = get_or_default(params, 'approx_res', false);

L   = get_or_default(factors, 'L', []);
D   = get_or_default(factors, 'D', []);
P   = get_or_default(factors, 'P', []);
e   = get_or_default(factors, 'e', []);
d   = get_or_default(factors, 'd', []);
rho = get_or_default(factors, 'rho', 0);

if isempty(L) || isempty(e) || isempty(d)
  L = []; D = []; P = []; e = []; d = [];
end

if rho == 0 || ~ada_rho
  rho = get_or_default(params, 'rho', 1.0);
end

% Initialize z^k, \tilde z^k and xi.
[m, n] = size(A);
x = zeros(n, 1);     xt = zeros(n, 1);
y = zeros(m, 1);     yt = zeros(m, 1);
z = zeros(n + m, 1); zt = zeros(n + m, 1);
xi = 1.0;

% Start timer.
if ~quiet
  total_time = tic;
end

% Normalize A
if isempty(e) || isempty(d)
  if norml
    [A, d, e] = equil(A, 2);
    ff = sqrt(norm(d) * sqrt(n) / (norm(e) * sqrt(m)));
    d = d / ff; e = e * ff;
  else
    d = ones(m, 1);
    e = ones(n, 1);
  end
else
  A = bsxfun(@times, bsxfun(@times, A, d), e');
end

if ~quiet
  fprintf('iter :\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\t%8s\n', ...
     'r', 'eps_pri', 's', 'eps_dual', 'gap', 'eps_gap', 'primal');
end

% Constants.
alpha = 1.7;
delta_min = 1.05;
delta = 1.05;
gamma = 1.01;
last_up = 0;
last_dn = 0;
kappa = 0.9;
tau = 0.8;

for iter = 0:MAXITR-1
  % Previous state varibles.
  xprev = x; yprev = y; zprev = z;
  
  % Evaluate proximal operators of f and g.
  %   y^{k+1/2} = prox(y^k - \tilde y^k)
  %   x^{k+1/2} = prox(x^k - \tilde x^k)
  y12 = eval_prox(prox_f, y - yt, rho, 1 ./ d);
  x12 = eval_prox(prox_g, x - xt, rho, e);
  z12 = [x12; y12];
  v12 = rho * (z - zt - z12);
  
  % Check stopping criteria.
  obj = obj_fn(x12 .* e, y12 ./ d);
  if approx_res
    eps_pri  = sqrt(m + n) * ABSTOL + RELTOL * norm(z12);
    eps_dual = sqrt(m + n) * ABSTOL + RELTOL * norm(v12);
    r = z12 - z;
    s = rho * (z - zprev);
  else
    eps_pri  = sqrt(m) * ABSTOL + RELTOL * norm(z12);
    eps_dual = sqrt(n) * ABSTOL + RELTOL * norm(v12);
    r = A * x12 - y12;
    s = A' * v12(n + 1:end) + v12(1:n);
  end
  eps_gap = sqrt(m + n) * ABSTOL + RELTOL * abs(obj);
  prires = norm(r);
  duares = norm(s);
  absgap = abs(v12' * z12);

  converged = iter > 1 && prires < eps_pri && duares < eps_dual && absgap < eps_gap;
  if ~quiet && (mod(iter, 10) == 0 || converged)
    primal = obj_fn(x12 .* e, y12 ./ d);
    fprintf('%4d :\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n', ...
        iter, prires, eps_pri, duares, eps_dual, absgap, eps_gap, primal);
  end

  if converged
    break
  end
  
  if ~quiet && iter == 0
    factor_time = tic;
  end
  
  % Project onto graph of {(x, y) \in R^{n + m} | y = Ax}, updating
  %   (x^{k+1}, y^{k+1}) = Pi_A(x^{k+1/2} + \tilde x^k, 
  %                             y^{k+1/2} + \tilde y^k)
  [z, L, D, P] = project_graph(z12, A, L, D, P);
  z = alpha * z + (1 - alpha) * zprev;

  if ~quiet && iter == 0
    factor_time = toc(factor_time);
  end
  
  x = z(1:n);
  y = z(n + 1:n + m);
  
  % Update dual variables.
  %   \tilde x^{k+1} = \tilde x^{k} + x^{k+1/2} - x^k
  %   \tilde y^{k+1} = \tilde y^{k} + y^{k+1/2} - y^k
  xt = xt + alpha * x12 + (1 - alpha) * xprev - x;
  yt = yt + alpha * y12 + (1 - alpha) * yprev - y;
  
  prires = norm(z - z12);
  duares = rho * norm(z - zprev);

  % Update rho
  if ada_rho
    if prires > xi * eps_pri && duares < xi * eps_dual && iter * tau > last_dn
      rho = rho * delta; xt = xt / delta; yt = yt / delta;
      delta = delta * gamma;
      last_up = iter;
    elseif prires < xi * eps_pri && duares > xi * eps_dual && iter * tau > last_up
      rho = rho / delta; xt = xt * delta; yt = yt * delta;
      delta = delta * gamma;
      last_dn = iter;
    elseif prires < xi * eps_pri && duares < xi * eps_dual
      xi = xi * kappa;
    else
      delta = max(delta / gamma, delta_min);
    end
  end

  zt = [xt; yt];
end

% Set factors for output.
factors.L = L;
factors.D = D;
factors.P = P;
factors.d = d;
factors.e = e;
factors.rho = rho;
n_iter = iter;

% Scale output
x = x12 .* e;
y = y12 ./ d;

if ~quiet
  fprintf('factorization time: %.2e seconds\n', factor_time);
  fprintf('total iterations: %d\n', iter);
  fprintf('total time: %.2f seconds\n', toc(total_time));
end

end

function y = eval_prox(f_prox, x, rho, d)
% Evaluates the proximal operator(s) of f on x. f_prox may either be a 
% function handle or a cell array of function handles.

if iscell(f_prox)
  y = nan(size(x));
  for i = 1:length(f_prox)
    y(i) = f_prox{i}(x(i) * d(i), rho / d(i) ^ 2) / d(i);
  end
else
  y = f_prox(x .* d, rho ./ d .^ 2) ./ d;
end

end

function [z, L, D, P] = project_graph(v, A, L, D, P)
% Project v onto the graph of A. This is equivalent to solving
%
%    minimize    (1/2) ||x - c||_2^2 + (1/2) ||y - d||_2^2,
%    subject to  y = Ax.
% 
% Supports factorization caching and both dense and sparse A.

[m, n] = size(A);
c = v(1:n);
d = v(n + 1:end);

if issparse(A)
  if isempty(P) || isempty(L) || isempty(D)
    % Solve KKT system.
    K = [speye(n) A' ; A -speye(m)];
    [L, D, P] = ldl(K);
  end

  z = P * (L' \ (D \ (L \ (P' * sparse([c + A' * d; zeros(m, 1)])))));
else
  % Project fat/skinny matrices onto the graph of A, by forming the normal
  % equations, and taking Cholesky decomposition.
  if m < n
    % Fat matrices.
    if isempty(L)
      L = chol(eye(m) + A * A');
    end
    y = d + L \ (L' \ (A * c - d));
    x = c + A' * (d - y);
  else
    % Skinny matrices. Use matrix inversion lemma.
    if isempty(L)
      L = chol(eye(n) + A' * A);
    end
    x = L \ (L' \ (c + A' * d));
    y = A * x;
  end

  z = [x; y];
end

end

function output = get_or_default(input, var, default)
% Returns input.var if var is a field in input, otherwise returns default.

if isfield(input, var)
  output = input.(var);
else
  output = default;
end

end

function [A, d, e] = equil(A, nrm)
[m, n] = size(A);

d = ones(m, 1);
e = ones(n, 1);

if m > n
  e = 1 ./ norms(A, nrm, 1)';
else
  d = 1 ./ norms(A, nrm, 2);
end

A = bsxfun(@times, bsxfun(@times, A, d), e');

end