function pogs_time = basis_pursuit(m, n, params)
%BASIS_PURSUIT

if nargin == 2
  params = [];
end

% Generate data.
rng(0, 'twister');

A = 1 / n * randn(m, n);
b = A * ((rand(n, 1) > 0.8) .* randn(n, 1)) + 0.5 * randn(m, 1);

f.h = kIndEq0;
f.b = b;
g.h = kAbs;

% Solve
tic
pogs(A, f, g, params);
pogs_time = toc;

end
