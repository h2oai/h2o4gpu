function [pogs_time, cvx_time] = lp_eq(m, n, params, comp_cvx, density)
%LP_EQ

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
rng(0, 'twister')

if density == 1
  A = 4 / n * rand(m, n);
else
  A = 4 / n * sprand(m, n, density);
end
b = A * rand(n, 1);
c = 1 / n * rand(n, 1);

f.h = [kIndEq0(m); kIdentity];
f.b = [b; 0];
g.h = kIndGe0;

% Solve with pogs.
if ~issparse(A)
  As = single([A; c']);
else
  As = [A; c'];
end
tic
[~, ~, ~, ~, status] = pogs(As, f, g, params);
pogs_time = toc;

if status > 0
  pogs_time = nan;
end

% Solve with CVX.
if comp_cvx
  tic
  cvx_begin quiet
    variable x(n)
    minimize(c' * x)
    A * x == b;
    x >= 0;
  cvx_end
  cvx_time = toc;
end

end

