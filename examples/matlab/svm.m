function [h2oaiglm_time, cvx_time] = svm(m, n, params, comp_cvx, density)
%SVM

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end
if nargin <= 4
  density = 1;
end

cvx_time = nan;

if mod(m, 2) ~= 0
  m = m + 1;
end

% Generate data.
rng(0, 'twister')

lambda = 1.0;
N = m / 2;

if density == 1
  x = 1 / sqrt(n) * [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)];
  y = [ones(N, 1); -ones(N, 1)];
  A = [(-y * ones(1, n)) .* x, -y];
else
  mu_plus  = sprandn(N, n, density);
  mu_minus = sprandn(N, n, density);
  x = 1 / sqrt(n) * [mu_plus  + 1. * (mu_plus  ~= 0)
                     mu_minus - 1. * (mu_minus ~= 0)];
  y = [ones(N, 1); -ones(N, 1)];
  A = sparse([(-y * ones(1, n)) .* x, -y]);
end

f.h = kMaxPos0;
f.b = -1;
f.c = lambda;
g.h = [kSquare(n); 0];

% Solve with h2oaiglm
if ~issparse(A)
  As = single(A);
else
  As = A;
end

tic
[~, ~, ~, ~, ~, status] = h2oaiglm(As, f, g, params);
h2oaiglm_time = toc;

if status > 0
  h2oaiglm_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n + 1)
    minimize(lambda * sum(max(A * x + 1, 0)) + 1 / 2 * (x' * x))
  cvx_end
  cvx_time = toc;
end

end
