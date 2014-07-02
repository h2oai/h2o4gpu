function pogs_time = entropy(m, n, params)
%ENTROPY

if nargin == 2
  params.rho = 1e3;
  params.rel_tol = 1e-6;
  params.abs_tol = 1e-6;
end

% Generate data.
rng(0, 'twister');

F = randn(m, n) / n;
b = F * rand(n, 1) + rand(m, 1);
A = [F; ones(1, n)];

f.h = [kIndLe0(m); kIndEq0];
f.b = [b; 1];
g.h = kNegEntr;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end
