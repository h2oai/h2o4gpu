function [pogs_time, cvx_time] = logistic(m, n, params, comp_cvx)
%LOGISTIC

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
  cvx_time = nan;
end

% Generate data.
rng(0, 'twister');

x = randn(n + 1, 1) / n .* (rand(n + 1, 1) > 0.8);
A = [rand(m, n), ones(m, 1)];
y = (rand(m, 1) < 1 ./ (1 + exp(-A * x)));

f.h = kLogistic;
f.d = -y;
g.h = kAbs;
g.c = 0.06 * norm(A' * (ones(m, 1) / 2 - y), inf);

% Solve with pogs
A = single(A);
tic
pogs(A, f, g, params);
pogs_time = toc;

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
