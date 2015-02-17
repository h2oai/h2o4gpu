function [pogs_time, cvx_time] = lasso(m, n, params, comp_cvx)
%LASSO

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
  cvx_time = nan;
end

% Generate data.
rng(0, 'twister');

A = 1 / n * rand(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.1 * randn(m, 1);
lambda = 1 / 3 * norm(A' * b, inf);

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve with pogs
A = single(A);
tic
pogs(A, f, g, params);
pogs_time = toc;

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

