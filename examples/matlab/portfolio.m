function [pogs_time, cvx_time] = portfolio(m, n, params, comp_cvx)
%PORTFOLIO

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
  cvx_time = nan;
end

% Generate data.
rng(0, 'twister');

A = [randn(n, m) ones(n, 1)]';
d = rand(n, 1);
r = -rand(n, 1);
gamma = 1;

f.h = kSquare;
f.c = gamma;
g.h = kIndGe0;
g.d = r;
g.e = gamma * d;

% Solve with pogs
As = single(A);
tic
pogs(As, f, g, params);
pogs_time = toc;

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

