function [pogs_time, cvx_time] = lp_ineq(m, n, params, comp_cvx, density)
%LP_INEQ

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
  A = -[4 / n * rand(m - n, n); eye(n)];
else
  A = -[4 / n * sprand(m, n, density); speye(n)];
end
b = A * rand(n, 1) + 0.2 * rand(m, 1);
c = 1 / n * rand(n, 1);

f.h = kIndLe0;
f.b = b;
g.h = kIdentity;
g.c = c;

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
    variable x(n)
    minimize(c' * x)
    A * x <= b;
  cvx_end
  cvx_time = toc;
end

end

