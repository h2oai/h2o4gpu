function pogs_time = lp_eq(m, n, params)
%LP_EQ

if nargin == 2
  params = [];
end

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister')

A = 4 / n * rand(m, n);
b = A * rand(n, 1);
c = rand(n, 1);

f.h = [kIndEq0 * ones(m, 1); kIdentity];
f.b = [b; 0];
g.h = kIndGe0;

% Solve.
tic
pogs([A; c'], f, g, params);
pogs_time = toc;

end

