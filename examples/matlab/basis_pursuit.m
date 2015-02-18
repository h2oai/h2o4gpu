function [pogs_time, cvx_time] = basis_pursuit(m, n, params, comp_cvx)
%BASIS_PURSUIT

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

A = randn(m, n);
b = A * ((rand(n, 1) > 0.5) .* randn(n, 1) / sqrt(n));

f.h = kIndEq0;
f.b = b;
g.h = kAbs;

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
    variables x(n)
    minimize(norm(x, 1))
    subject to
      A * x == b;
  cvx_end
  cvx_time = toc;
end

end
