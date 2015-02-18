function [pogs_time, cvx_time] = huber_fit(m, n, params, comp_cvx)
%HUBER_FIT

if nargin <= 2
  params = [];
  params.rho = 1e2;
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

A = randn(m, n);
x_true = randn(n, 1) / sqrt(n);
b = A * x_true + 10 * rand(m, 1) .* (rand(m, 1) > 0.95);

f.h = kHuber;
f.b = b;
g.h = kZero;

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
    minimize(sum(huber(A * x - b)))
  cvx_end
  cvx_time = toc;
end

end
