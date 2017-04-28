function [h2oaiglm_time, cvx_time] = lp_cone(m, n, params, comp_cvx, density)
%LP_CONE

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

N = floor(min(m, n) / 4);

if density == 1
  A = 4 / n * rand(m, n);
else
  A = 4 / n * sprand(m, n, density);
end
b = A * rand(n, 1) + [zeros(N, 1); 0.1 * rand(m - N, 1)];
c = -A' * rand(m ,1);

% Solve with h2oaiglm.
if ~issparse(A)
  As = single(A);
else
  As = A;
end

f.h = [kIndEq0(N); kIndLe0(m - N)];
f.b = b;
g.h = kIdentity;
g.a = c;

tic
[~, ~, ~, ~, ~, status] = h2oaiglm(As, f, g, params);
h2oaiglm_time = toc;

if status > 0
  h2oaiglm_time = nan;
end

% Solve with CVX.
if comp_cvx
  tic
  cvx_begin quiet
    variables x(n) s(m)
    minimize(c' * x)
    A * x + s == b;
    s(1:N) == 0;
    s(N + 1:m) >= 0;
  cvx_end
  cvx_time = toc;
end

end
