function [h2ogpuml_time, cvx_time] = basis_pursuit(m, n, params, comp_cvx, density)
%BASIS_PURSUIT

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
b = A * ((rand(n, 1) > 0.5) .* randn(n, 1) / sqrt(n));

f.h = kIndEq0;
f.b = b;
g.h = kAbs;

% Solve with h2ogpuml
if ~issparse(A)
  As = single(A);
else
  As = A;
end
tic
[~, ~, ~, ~, ~, status] = h2ogpuml(full(As), f, g, params);
h2ogpuml_time = toc;

if status > 0
  h2ogpuml_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n)
    minimize(norm(x, 1))
    subject to
      A * x == b;
  cvx_end
  cvx_time = toc;
end

end
