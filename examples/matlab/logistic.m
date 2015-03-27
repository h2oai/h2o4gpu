function [pogs_time, cvx_time] = logistic(m, n, params, comp_cvx, density)
%LOGISTIC

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
A = [A, ones(m, 1)];
x_true = randn(n + 1, 1) .* [rand(n, 1) > 0.8; 1];
y = (rand(m, 1) < 1 ./ (1 + exp(-A * x_true)));

f.h = kLogistic;
f.d = -y;
g.h = kAbs;
g.c = 0.05 * norm(A' * (ones(m, 1) / 2 - y), inf);

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
    variables x(n + 1)
    minimize(sum(log(1 + exp(A * x))) + g.c * norm(x, 1))
  cvx_end
  cvx_time = toc;
end

end
