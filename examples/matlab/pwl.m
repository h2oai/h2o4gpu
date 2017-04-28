function [h2oaiglm_time, cvx_time] = pwl(m, n, params, comp_cvx, density)
%PWL

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
  A = [1 / n * rand(m-n, n); -eye(n)];
else
  A = [1 / n * sprand(m - n, n); -speye(n)];
end

b = A * rand(n, 1) + 2 * randn(m, 1);
A = [A -ones(m, 1)];

f.h = kIndLe0;
f.b = b;
g.h = [kZero(n); kIdentity];

% Solve with h2oaiglm
if ~issparse(A)
  As = single(A);
else
  As = A;
end
tic
[~, ~, ~, ~, ~, status] = h2oaiglm(As, f, g, params);
h2oaiglm_time = toc;

if status > 0
  h2oaiglm_time = nan;
end

% Solve with CVX
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n)
    minimize(max(A * x - b))
  cvx_end
  cvx_time = toc;
end

end

