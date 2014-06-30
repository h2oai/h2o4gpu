function pogs_time = lasso(m, n, params)
%LASSO

if nargin == 2
  params = [];
end

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister');

A = 1 / n * rand(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.1 * randn(m, 1);
lambda = 1 / 2 * norm(A' * b, inf);

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

