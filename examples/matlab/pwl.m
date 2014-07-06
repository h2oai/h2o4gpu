function pogs_time = pwl(m, n, params)
%PWL

if nargin == 2
  params = [];
  params.rho = 1e-3;
end

% Generate data.
rng(0, 'twister');

A = [1 / n * rand(m-n, n); -eye(n)];
b = A * rand(n, 1) + 2 * randn(m, 1);
A = [A -ones(m, 1)];

f.h = kIndLe0;
f.b = b;
g.h = [kZero(n); kIdentity];

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

