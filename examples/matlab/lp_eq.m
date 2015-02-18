function [pogs_time, cvx_time] = lp_eq(m, n, params, comp_cvx)
%LP_EQ

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister')

A = 4 / n * rand(m, n);
b = A * rand(n, 1);
c = 1 / n * rand(n, 1);

f.h = [kIndEq0(m); kIdentity];
f.b = [b; 0];
g.h = kIndGe0;

% Solve with pogs.
As = single([A; c']);
tic
[~, ~, ~, ~, status] = pogs(As, f, g, params);
pogs_time = toc;

if status > 0
  pogs_time = nan;
end

% Solve with CVX.
if comp_cvx
  tic
  cvx_begin
    variable x(n)
    minimize(c' * x)
    A * x == b;
    x >= 0;
  cvx_end
  cvx_time = toc;
end

end

