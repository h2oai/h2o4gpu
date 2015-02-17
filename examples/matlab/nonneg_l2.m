function [pogs_time, cvx_time] = nonneg_l2(m, n, params, comp_cvx)
%NONNEG_L2

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
  cvx_time = nan;
end

% Generate data.
rng(0, 'twister')

n_half = floor(0.9 * n);
A = 2 / n * rand(m, n);
b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1);

f.h = kSquare;
f.b = b;
g.h = kIndGe0;

% Solve with pogs
As = single(A);
tic
pogs(As, f, g, params);
pogs_time = toc;

% Solve with CVX
if comp_cvx
  tic
  cvx_begin
    variables x(n)
    minimize(norm(A * x - b))
    x >= 0;
  cvx_end
  cvx_time = toc;
end

end

