function pogs_time = lp_eq(m, n, nnz, params)
%LP_EQ

if nargin == 2
  params = [];
end

% Generate data.
rng(0, 'twister')

A = 4 / n * sprand(m, n, nnz / (m * n));
b = A * rand(n, 1);
c = rand(n, 1);

f.h = [kIndEq0(m); kIdentity];
f.b = [b; 0];
g.h = kIndGe0;

% Solve.
tic
x_pogs = pogs([A; c'], f, g, params);
pogs_time = toc;

cvx_begin
variables x(n)
minimize(c' * x)
subject to
  b == A * x
  x >= 0
cvx_end

(c' * x_pogs - cvx_optval) / (cvx_optval)


end

