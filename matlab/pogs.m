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
RELTOL = get_or_default(params, 'RELTOL', 1e-2);
MAXITR = get_or_default(params, 'MAXITR', 10000);
rho    = get_or_default(params, 'rho', 1.0);
quiet  = get_or_default(params, 'quiet', false);
norml  = get_or_default(params, 'norml', true);

L  = get_or_default(factors, 'L', []);
D  = get_or_default(factors, 'D', []);
P  = get_or_default(factors, 'P', []);
AA = get_or_default(factors, 'AA', []);
e  = get_or_default(factors, 'e', []);
d  = get_or_default(factors, 'd', []);

if isempty(L) || isempty(AA) || isempty(e) || isempty(d)
  L = []; D = []; P = []; AA = []; e = []; d = [];
end

% Initialize x^k and \tilde x^k.
[m, n] = size(A);
x = zeros(n, 1);     xt = zeros(n, 1);
y = zeros(m, 1);     yt = zeros(m, 1);
z = zeros(n + m, 1); zt = zeros(n + m, 1);

% Start timer.
if ~quiet
  total_time = tic;
end

% Normalize A
if isempty(e) || isempty(d)
  if norml 
    [A, d, e] = sk_equil(A);
    if m < n
      sms = sqrt(mean(sum(A.^2,2)));
    else
      sms = sqrt(mean(sum(A.^2,1)));
    end
    d = d / sqrt(sms);
    e = e / sqrt(sms);
    A = A / sms;
  else
    d = ones(m, 1);
    e = ones(n, 1);
  end
else
  A = bsxfun(@times, bsxfun(@times, A, d), e');
end

% Precompute AAt or AtA.
if isempty(AA) && ~issparse(A)
  if m < n
    AA = A * A';
  else
    AA = A' * A;
  end
end

if ~quiet
  fprintf('iter :\t%8s\t%8s\t%8s\t%8s\t%8s\n', 'r', 'eps_pri', 's', ...
      'eps_dual', 'objective');
end

for iter = 0:MAXITR-1
  % Evaluate proximal operators of f and g.
  %   y^{k+1/2} = prox(y^k - \tilde y^k)
  %   x^{k+1/2} = prox(x^k - \tilde x^k)
  y12 = eval_prox(prox_f, y - yt, rho, 1 ./ d);
  x12 = eval_prox(prox_g, x - xt, rho, e);
  z12 = [x12; y12];

  zprev = z; 

  if ~quiet && iter == 0
    factor_time = tic;
  end

  % Project onto graph of {(x, y) \in R^{n + m} | y = Ax}, updating
  %   (x^{k+1}, y^{k+1}) = Pi_A(x^{k+1/2} + \tilde x^k, 
  %                             y^{k+1/2} + \tilde y^k)
  [z, L, D, P] = project_graph(z12 + zt, A, AA, L, D, P);

  if ~quiet && iter == 0
    factor_time = toc(factor_time);
  end

  x = z(1:n);
  y = z(n + 1:n + m);

  % Check stopping ckriteria.
  eps_pri  = sqrt(n) * ABSTOL + RELTOL * max(norm(z12), norm(z));
  eps_dual = sqrt(n) * ABSTOL + RELTOL * norm(rho * zt);
  prires = norm(z12 - z);
  duares = rho * norm(z - zprev);

  converged = iter > 1 && prires < eps_pri && duares < eps_dual;
  if ~quiet && (mod(iter, 1) == 0 || converged)
    fprintf('%4d :\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n', ...
        iter, prires, eps_pri, duares, eps_dual, obj_fn(x .* e, y ./ d));
  end
  
  if norm(z12 - z) > 10 * norm(z - zprev)
    rho = rho * 10; xt = xt / 10; yt = yt / 10;
  elseif norm(z12 - z) < 0.1 * norm(z - zprev)
    rho = rho / 10; xt = xt * 10; yt = yt * 10;
  end

  if converged
    break
  end
  
  % Update dual variables.
  %   \tilde x^{k+1} = \tilde x^{k} + x^{k+1/2} - x^k
  %   \tilde y^{k+1} = \tilde y^{k} + y^{k+1/2} - y^k
  xt = xt + x12 - x;
  yt = yt + y12 - y;
  zt = [xt; yt];
end

% Set factors for output.
factors.L = L;
factors.D = D;
factors.P = P;
factors.AA = AA;
factors.d = d;
factors.e = e;
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

function [z, L, D, P] = project_graph(v, A, AA, L, D, P)
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
    K = [ speye(n) A' ; A -speye(m) ];
    [L, D, P] = ldl(K);
  end

  z = P * (L' \ (D \ (L \ (P' * sparse([ c + A' * d ; zeros(m, 1) ])))));
else
  % Project fat/skinny matrices onto the graph of A, by forming the normal
  % equations, and taking Cholesky decomposition.
  if m < n
    % Fat matrices.
    if isempty(L)
      L = chol(eye(m) + AA);
    end
    y = L \ (L' \ (A * c + AA * d));
    x = c + A' * (d - y);
  else
    % Skinny matrices. Use matrix inversion lemma.
    if isempty(L)
      L = chol(eye(n) + AA);
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

function [A, d, e] = sk_equil(A)

A1 = abs(A);

max_it = 10;

d = ones(size(A1, 1), 1);
e = ones(size(A1, 2), 1);

for i = 1:max_it
  e = 1 ./ (A1' * d);
  d = 1 ./ (A1 * e);
  nd = norm(d) / sqrt(size(A1, 1));
  ne = norm(e) / sqrt(size(A1, 2));
  d = d * sqrt(ne / nd);
  e = e * sqrt(nd / ne);
end

A = bsxfun(@times, bsxfun(@times, A, d), e');

end
