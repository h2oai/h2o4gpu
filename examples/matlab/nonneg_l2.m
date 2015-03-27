function [pogs_time, cvx_time] = nonneg_l2(m, n, params, comp_cvx, density)
%NONNEG_L2

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
rng(0, 'twister')

if density == 1
  A = randn(m, n);
else
  A = sprandn(m, n, density);
end
x_true = 2 * rand(n, 1) / sqrt(n);
b = A * (x_true + 0.5 * randn(n, 1) / sqrt(n)) + 0.3 * randn(m, 1);

f.h = kSquare;
f.b = b;
g.h = kIndGe0;

% Solve with pogs
if ~issparse(A)
  As = single(A);
else
  As = A;
end
tic
[~, ~, ~, ~, ~, status] = pogs(As, f, g, params);
pogs_time = toc;

if status > 0
  pogs_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n)
    minimize(norm(A * x - b))
    x >= 0;
  cvx_end
  cvx_time = toc;
end

end

