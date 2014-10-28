function pogs_time = lasso_sp(m, n, nnz, params)
%LASSO

if nargin == 3
%   params = [];
  params.rel_tol = 1e-4;
  params.abs_tol = 1e-5;
end

% Generate data.
rng(0, 'twister');

A = sprand(m, n, nnz / (m * n));
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1));
b = b + 0.1 * norm(b) * randn(m, 1);
lambda = 1 / 3 * norm(A' * b, inf);

f.h = kSquare;
f.b = b;
g.h = kAbs;
g.c = lambda;

% Solve
tic
x_pogs = pogs(A, f, g, params);
pogs_time = toc;

cvx_begin
variables x(n) y(m)
minimize(1/2*(y-b)'*(y-b) + lambda * norm(x,1))
subject to
  y == A * x
cvx_end

(1/2*norm(A * x_pogs - b, 2)^2 + lambda * norm(x_pogs,1) - cvx_optval) / (cvx_optval)

end

