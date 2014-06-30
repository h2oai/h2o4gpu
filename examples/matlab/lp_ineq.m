function pogs_time = lp_ineq(m, n, params)
%LP_INEQ

if nargin == 2
  params = [];
end

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister');

A = -[4 / n * rand(m - n, n); eye(n)];
b = A * rand(n, 1) + 0.2 * rand(m, 1);
c = rand(n, 1);

f.h = kIndLe0;
f.b = b;
g.h = kIdentity;
g.c = c;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end

