function [pogs_time, cvx_time] = lp_ineq(m, n, params, comp_cvx)
%LP_INEQ

if nargin <= 2
  params = [];
end
if nargin <= 3
  cvx_time = nan;
  comp_cvx = false;
end

% Generate data.
rng(0, 'twister');

A = -[4 / n * rand(m - n, n); eye(n)];
b = A * rand(n, 1) + 0.2 * rand(m, 1);
c = 1 / n * rand(n, 1);

f.h = kIndLe0;
f.b = b;
g.h = kIdentity;
g.c = c;

% Solve with pogs
A = single(A);
tic
x = pogs(A, f, g, params);
pogs_time = toc;

% Solve with CVX
if comp_cvx
  tic
  cvx_begin
    variable x(n)
    minimize(c' * x)
    A * x <= b;
  cvx_end
  cvx_time = toc;
end

end

