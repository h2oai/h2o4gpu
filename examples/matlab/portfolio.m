function [pogs_time, cvx_time] = portfolio(m, n, params, comp_cvx, density)
%PORTFOLIO

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
  A = [randn(m, n) ones(n, 1)];
else
  A = [sprandn(m, n, density) ones(n, 1)];
end
d = rand(n, 1);
r = -rand(n, 1);
gamma = 1;

f.h = kSquare;
f.c = gamma;
g.h = kIndGe0;
g.d = r;
g.e = gamma * d;

% Solve with pogs
if ~issparse(A)
  As = single(A);
else
  As = A;
end
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
    variables x(n) y(m + 1)
    minimize(r' * x + sum(d .* x .* x) + gamma * sum_square(y))
    y == A * x;
    x >= 0;
  cvx_end
  cvx_time = toc;
end

end

