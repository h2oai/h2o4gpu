function [h2ogpuml_time, cvx_time] = lasso(m, n, params, comp_cvx, density)
%LASSO

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

% Generate data.
rng(0, 'twister');

if density == 1
  A = randn(m, n);
else
  A = sprandn(m, n, density);
end
x_true = (randn(n, 1) > 0.8) .* randn(n, 1) / sqrt(n);
b = A * x_true + 0.5 * randn(m, 1);
lambda = 0.2 * norm(A' * b, inf);

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve with h2ogpuml
if ~issparse(A)
  As = single(A);
else
  As = A;
end
tic
[~, ~, ~, ~, ~, status] = h2ogpuml(As, f, g, params);
h2ogpuml_time = toc;

if status > 0
  h2ogpuml_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n)
    minimize(sum_square(A * x - b) + lambda * norm(x, 1))
  cvx_end
  cvx_time = toc;
end

end

