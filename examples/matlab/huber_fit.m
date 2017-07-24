function [h2ogpuml_time, cvx_time] = huber_fit(m, n, params, comp_cvx, density)
%HUBER_FIT

if nargin <= 2
  params = [];
  params.rho = 1e2;
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
x_true = randn(n, 1) / sqrt(n);
b = A * x_true + 10 * rand(m, 1) .* (rand(m, 1) > 0.95);

f.h = kHuber;
f.b = b;
g.h = kZero;

% Solve with h2ogpuml
if ~issparse(A)
  As = double(A);
else
  As = A;
end
tic
[~, ~, ~, ~, ~, status] = h2ogpuml(As, f, g, params);
h2ogpuml_time = toc;

if status > 0
  h2ogpuml_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n) y(m)
    dual variable lambda
    minimize(sum(huber(y - b)))
      lambda : y == A * x
  cvx_end
  cvx_time = toc;
end

end
