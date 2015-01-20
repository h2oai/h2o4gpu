function pogs_time = entropy(m, n, params)
%ENTROPY

if nargin == 2
  params.rho = 1;
  params.rel_tol = 1e-3;
  params.abs_tol = 1e-4;
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
x = pogs(A, f, g, params);
pogs_time = toc;

end
