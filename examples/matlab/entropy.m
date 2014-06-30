function pogs_time = entropy(m, n, params)
%ENTROPY

if nargin == 2
  params.rho = 1e4;
  params.rel_tol = 1e-6;
  params.abs_tol = 1e-6;
end

% Load function definitions.
pogs_defs

% Generate data.
rng(0, 'twister');

F = randn(m, n) / n;
b = F * rand(n, 1) + rand(m, 1);
A = [F; ones(1, n)];

f.h = [kIndLe0 * ones(m, 1); kIndEq0];
f.b = [b; 1];
g.h = kNegEntr;

% Solve
tic
x_pogs = pogs(A, f, g, params);
pogs_time = toc;

cvx_begin
    variable x(n)
    minimize(-sum(entr(x)))
    F * x <= b
    sum(x) == 1
cvx_end

end
