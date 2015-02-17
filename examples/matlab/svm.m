function [pogs_time, cvx_time] = svm(m, n, params, comp_cvx)
%SVM

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
  cvx_time = nan;
end

if mod(m, 2) ~= 0
  m = m + 1;
end

% Generate data.
rng(0, 'twister')

lambda = 1.0;
N = m / 2;

x = 1 / n * [randn(N, n) + ones(N, n); randn(N, n) - ones(N, n)];
y = [ones(N, 1); -ones(N, 1)];
A = [(-y * ones(1, n)) .* x, -y];

f.h = kMaxPos0;
f.b = -1;
f.c = lambda;
g.h = [kSquare(n); 0];

% Solve with pogs
As = single(A);
tic
pogs(As, f, g, params);
pogs_time = toc;

% Solve with CVX
if comp_cvx
  tic
  cvx_begin
    variables x(n + 1)
    minimize(lambda * sum(max(A * x + 1, 0)) + 1 / 2 * (x' * x))
  cvx_end
  cvx_time = toc;
end

end