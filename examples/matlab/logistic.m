function [pogs_time, cvx_time] = logistic(m, n, params, comp_cvx)
%LOGISTIC

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

x_true = randn(n + 1, 1) .* [rand(n, 1) > 0.8; 1];
A = [randn(m, n), ones(m, 1)];
y = (rand(m, 1) < 1 ./ (1 + exp(-A * x_true)));

f.h = kLogistic;
f.d = -y;
g.h = kAbs;
g.c = 0.05 * norm(A' * (ones(m, 1) / 2 - y), inf);

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
    variables x(n + 1)
    minimize(sum(log(1 + exp(A * x))) + g.c * norm(x, 1))
  cvx_end
  cvx_time = toc;
end

end
