function [h2ogpuml_time, cvx_time] = entropy(m, n, params, cvx_comp, density)
%ENTROPY

if nargin <= 2
  params = [];
end
if nargin <= 3
  cvx_comp = false;
end
if nargin <= 4
  density = 1;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

if density == 1
  F = randn(m, n) * sqrt(n);
else
  F = sprandn(m, n, density) * sqrt(n);
end
x_p = rand(n, 1);
x_p = x_p / sum(x_p);
b = F * x_p + 0.5 * rand(m, 1);
A = [F; ones(1, n)];

f.h = [kIndLe0(m); kIndEq0];
f.b = [b; 1];
g.h = kNegEntr;

% Solve with h2ogpuml
if ~issparse(A)
  As = single(A);
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
if cvx_comp
  tic
  cvx_begin quiet
    variables x(n)
    minimize(-sum(entr(x)))
    subject to
      F * x <= b;
      sum(x) == 1;
  cvx_end
  cvx_time = toc;
end

end
