function [pogs_time, cvx_time] = pwl(m, n, params, comp_cvx)
%PWL

if nargin <= 2
  params = [];
end
if nargin <= 3
  comp_cvx = false;
end

cvx_time = nan;

% Generate data.
rng(0, 'twister');

A = [1 / n * rand(m-n, n); -eye(n)];
b = A * rand(n, 1) + 2 * randn(m, 1);
A = [A -ones(m, 1)];

f.h = kIndLe0;
f.b = b;
g.h = [kZero(n); kIdentity];

% Solve with pogs
A = single(A);
tic
pogs(A, f, g, params);
pogs_time = toc;

% Solve with CVX
if comp_cvx
  tic
  cvx_begin
    variables x(n)
    minimize(max(A * x - b))
  cvx_end
  cvx_time = toc;
end

end

