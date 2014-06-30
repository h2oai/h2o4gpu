function pogs_time = nonneg_l2(m, n, params)
%NONNEG_L2

if nargin == 2
  params = [];
end

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister')

n_half = floor(0.9 * n);
A = 2 / n * rand(m, n);
b = A * [ones(n_half, 1); -ones(n - n_half, 1)] + 0.1 * randn(m, 1);

f.h = kSquare;
f.b = b;
g.h = kIndGe0;

% Solve.
tic
pogs(A, f, g, params);
pogs_time = toc;

end

