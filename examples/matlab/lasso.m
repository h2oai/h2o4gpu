function [pogs_time, cvx_time] = lasso(m, n, params, comp_cvx)
%LASSO

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

A = randn(m, n);
x_true = (randn(n, 1) > 0.8) .* randn(n, 1) / sqrt(n);
b = A * x_true + 0.5 * randn(m, 1);
lambda = 0.2 * norm(A' * b, inf);

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve with pogs
As = single(A);
tic
[~, ~, ~, ~, status] = pogs(As, f, g, params);
pogs_time = toc;

if status > 0
  pogs_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin
    variables x(n)
    minimize(sum_square(A * x - b) + lambda * norm(x, 1))
  cvx_end
  cvx_time = toc;
end

end

